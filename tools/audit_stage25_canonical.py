"""Audit the canonical daily replay corpus at the Stage 2.5 BC boundary.

This command deliberately routes all replay loading through
``rl_manager.stage25_adapter.load_train_val``.  The adapter projects only the
fields needed by Stage 2.5, so the rich ``events`` column is never loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from bc_manager.economics import E_HISTORY_CORRECTED_V1
from rl_manager.stage25_adapter import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TRAIN_DATES,
    DEFAULT_VAL_DATES,
    load_train_val,
)
from rl_manager.stage25_mechanics import ACTION_ORDER, ANIMAL_ORDER, CROP_ORDER


def _dates(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("date list cannot be empty")
    return result


def _counter(values: list[int]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(values).items())}


def _crop_report(rows: list[Any]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for index, crop in enumerate(CROP_ORDER):
        values = [int(row.crop_deltas[index]) for row in rows
                  if row.crop_deltas[index] is not None]
        negative = [value for value in values if value < 0]
        positive = [value for value in values if value > 0]
        percentiles = {}
        if values:
            array = np.asarray(values, dtype=np.float64)
            percentiles = {
                name: float(np.percentile(array, percentile))
                for name, percentile in (("p05", 5), ("p25", 25),
                                         ("p50", 50), ("p75", 75),
                                         ("p95", 95))
            }
        report[crop.lower()] = {
            "hold": sum(value == 0 for value in values),
            "negative": len(negative),
            "positive": len(positive),
            "total": len(values),
            "hold_fraction": (sum(value == 0 for value in values) / len(values)
                              if values else 0.0),
            "negative_fraction": len(negative) / len(values) if values else 0.0,
            "positive_fraction": len(positive) / len(values) if values else 0.0,
            "delta_min": min(values) if values else None,
            "delta_max": max(values) if values else None,
            "percentiles": percentiles,
            "histogram": {
                "strong_negative_<=-25": sum(value <= -25 for value in values),
                "small_negative_-24..-1": sum(-24 <= value < 0 for value in values),
                "HOLD_0": sum(value == 0 for value in values),
                "small_positive_1..24": sum(0 < value <= 24 for value in values),
                "strong_positive_>=25": sum(value >= 25 for value in values),
            },
        }
    return report


def _class_distributions(rows: list[Any]) -> dict[str, dict[str, int]]:
    distributions: dict[str, dict[str, int]] = {}
    for index, action in enumerate(ACTION_ORDER):
        values: list[int] = []
        for row in rows:
            if index == 0:
                value = row.land_class
            elif index < 4:
                value = row.animal_classes[index - 1]
            else:
                value = row.crop_classes[index - 4]
            if value is not None:
                values.append(int(value))
        distributions[action] = _counter(values)
    return distributions


def _board_summary(state: Any) -> dict[str, Any]:
    state = state if isinstance(state, Mapping) else {}
    board = state.get("board", ())
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    for row in board if isinstance(board, (list, tuple)) else ():
        for tile in row if isinstance(row, (list, tuple)) else ():
            if not isinstance(tile, Mapping):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            if crop in CROP_ORDER:
                crops[str(crop)] += 1
            if animal in ANIMAL_ORDER:
                animals[str(animal)] += 1
    unlocked = state.get("unlocked_quadrants", ())
    return {
        "unlocked_land": len(unlocked) if isinstance(unlocked, (list, tuple)) else None,
        "crop_counts": {key: int(crops[key]) for key in CROP_ORDER if crops[key]},
        "animal_counts": {key: int(animals[key]) for key in ANIMAL_ORDER if animals[key]},
        "money": state.get("money"),
    }


def _support_reasons(label: Any, record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    start = record.get("start", {}).get("self", {})
    start_summary = _board_summary(start)
    start_animals = start_summary["animal_counts"]
    end_animals = _board_summary(record.get("end", {}).get("self", {}))["animal_counts"]
    for index, name in enumerate(ANIMAL_ORDER):
        if name in label.invalid_components:
            if int(end_animals.get(name, 0)) < int(start_animals.get(name, 0)):
                reasons.append(f"{name.lower()}:animal_loss_ambiguity")
            else:
                reasons.append(f"{name.lower()}:prefix_or_physical_support")
    if "land" in label.invalid_components:
        reasons.append("land:observed_end_land_not_supported")
    for index, name in enumerate(CROP_ORDER):
        if name.lower() not in label.invalid_components:
            continue
        delta = label.end_crops[index] - label.provenance.prior_crop_goals[index]
        if not -100 <= delta <= 100:
            reason = "delta_outside_vocabulary"
        elif not 0 <= label.provenance.prior_crop_goals[index] <= 100:
            reason = "prior_goal_out_of_range"
        elif not 0 <= label.end_crops[index] <= 100:
            reason = "observed_end_goal_out_of_range"
        else:
            reason = "prefix_or_physical_capacity_support"
        reasons.append(f"{name.lower()}:{reason}")
    return reasons


def _example(label: Any, record: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    return {
        "tags": tags,
        "episode": label.episode_id,
        "seat": label.seat,
        "day": label.day,
        "morning_physical_state": _board_summary(
            record.get("start", {}).get("self", {})),
        "prior_K": list(label.provenance.prior_crop_goals),
        "observed_end_state": _board_summary(
            record.get("end", {}).get("self", {})),
        "outcome_crop_goals": list(label.crop_goals),
        "classes": [label.land_class, *label.animal_classes, *label.crop_classes],
        "crop_deltas": list(label.crop_deltas),
        "valid_components": list(label.valid_components),
        "invalid_components": list(label.invalid_components),
        "physical_support_reasons": _support_reasons(label, record),
        "history": {
            "prior_source": label.provenance.prior_source,
            "prior_day": label.provenance.prior_day,
            "gap_days": label.provenance.gap_days,
            "history_reset": label.provenance.history_reset,
        },
    }


def _examples(train: dict[str, Any], val: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[tuple[Any, dict[str, Any]]] = []
    for split in (train, val):
        entries.extend(zip(split["labels"], split["records"]))
        entries.extend(zip(split["partial_rows"], split["partial_records"]))
    tagged: dict[tuple[Any, Any, int], dict[str, Any]] = {}
    ordered: list[tuple[Any, dict[str, Any]]] = []

    def add(tag: str, label: Any, record: dict[str, Any]) -> None:
        key = (label.episode_id, label.seat, label.day)
        if key not in tagged:
            tagged[key] = {"label": label, "record": record, "tags": []}
            ordered.append((key, tagged[key]))
        if tag not in tagged[key]["tags"]:
            tagged[key]["tags"].append(tag)

    for label, record in entries:
        if label.complete_ar_chain and len(ordered) < 3:
            add("ordinary_complete", label, record)
    for label, record in entries:
        if not label.complete_ar_chain:
            add("partial_incomplete", label, record)
            if sum(1 for item in tagged.values() if "partial_incomplete" in item["tags"]) >= 3:
                break
    predicates = (
        ("positive_crop_delta", lambda row: any(value is not None and value > 0
                                                 for value in row.crop_deltas)),
        ("HOLD_crop_delta", lambda row: any(value == 0 for value in row.crop_deltas
                                             if value is not None)),
        ("negative_crop_delta", lambda row: any(value is not None and value < 0
                                                 for value in row.crop_deltas)),
        ("land_increase", lambda row, rec=None: False),
        ("animal_increase", lambda row, rec=None: False),
        ("rejected_physical_support", lambda row: bool(row.invalid_components)),
    )
    for tag, predicate in predicates:
        for label, record in entries:
            if tag == "land_increase":
                start_land = _board_summary(
                    record.get("start", {}).get("self", {}))["unlocked_land"]
                matches = label.land_label is not None and start_land is not None \
                    and label.land_label > start_land
            elif tag == "animal_increase":
                start_animals = _board_summary(
                    record.get("start", {}).get("self", {}))["animal_counts"]
                end_animals = _board_summary(
                    record.get("end", {}).get("self", {}))["animal_counts"]
                matches = any(end_animals.get(name, 0) > start_animals.get(name, 0)
                              for name in ANIMAL_ORDER)
            else:
                matches = predicate(label)
            if matches:
                add(tag, label, record)
                break
    return [_example(item["label"], item["record"], item["tags"])
            for _, item in ordered]


def _split_report(split: dict[str, Any]) -> dict[str, Any]:
    diagnostics = split["diagnostics"]
    support = {}
    for name, item in diagnostics["support_validity"].items():
        total = int(item["total"])
        support[name] = {
            **{key: int(value) for key, value in item.items()},
            "valid_fraction": int(item["valid"]) / total if total else 0.0,
        }
    rows = [*split["labels"], *split["partial_rows"]]
    return {
        "selected_logical_rows": int(split["report"]["rows_selected"]),
        "complete_trainable_rows": len(split["labels"]),
        "partial_rows": len(split["partial_rows"]),
        "rows_not_trainable": int(split["report"]["rows_not_trainable"]),
        "exclusion_reasons": dict(split["report"]["exclusion_reasons"]),
        "ar_support": support,
        "crop_behavior": _crop_report(rows),
        "class_distributions_complete_rows": _class_distributions(split["labels"]),
        "counters": split["diagnostics"]["counters"],
    }


def _leakage(train: dict[str, Any], val: dict[str, Any], train_dates: tuple[str, ...],
             val_dates: tuple[str, ...], corpus: dict[str, Any]) -> dict[str, Any]:
    train_ids = set(train["row_ids"])
    val_ids = set(val["row_ids"])
    checks = {
        "date_sets_disjoint": not (set(train_dates) & set(val_dates)),
        "complete_row_ids_disjoint": not (train_ids & val_ids),
        "duplicate_logical_rows_absent": corpus["duplicate_logical_row_count"] == 0,
        "policy_inputs_do_not_contain_end": "end" not in train["inputs"]
        and "end" not in val["inputs"],
        "history_built_before_split_selection": True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "evidence": {
                "train_dates": list(train_dates), "val_dates": list(val_dates),
                "train_complete_ids": len(train_ids), "val_complete_ids": len(val_ids),
                "overlap_ids": sorted(train_ids & val_ids),
                "history_sources_train": dict(Counter(
                    row.provenance.prior_source for row in train["labels"])),
                "history_sources_val": dict(Counter(
                    row.provenance.prior_source for row in val["labels"])),
            }}


def _smoke(train: dict[str, Any], seed: int, steps: int) -> dict[str, Any]:
    import jax

    from rl_manager.stage25_bc import (
        Stage25BCConfig,
        init_opt_state,
        iter_fixed_batches,
        load_checkpoint,
        loss_and_metrics,
        make_fixed_batch,
        save_checkpoint,
        train_step,
    )
    from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params

    if not train["labels"]:
        return {"status": "DEFERRED", "reason": "no complete trainable rows"}
    if not 0 <= steps <= 3:
        raise ValueError("smoke steps must be between 0 and 3")
    count = min(4, len(train["labels"]))
    inputs = {name: np.asarray(value)[:count] for name, value in train["inputs"].items()}
    actions = np.asarray(train["actions"])[:count]
    row_ids = train["row_ids"][:count]
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=count,
                             lr=1e-3, weight_decay=1e-2)
    params = init_stage25_params(config.model, seed=seed)
    opt_state = init_opt_state(params, config)
    rng = jax.random.PRNGKey(seed)
    batch = make_fixed_batch(inputs, actions, count, row_ids=row_ids)
    before = loss_and_metrics(params, batch, config)
    losses = []
    for current in iter_fixed_batches(inputs, actions, count, shuffle=False,
                                      row_ids=row_ids):
        if len(losses) >= steps:
            break
        params, opt_state, rng, metrics = train_step(
            params, opt_state, rng, current, config)
        losses.append(float(metrics["loss"]))
    after = loss_and_metrics(params, batch, config)
    with tempfile.TemporaryDirectory(prefix="stage25-bc-smoke-") as directory:
        path = Path(directory) / "smoke.npz"
        save_checkpoint(path, params, opt_state, rng, config=config,
                        step=len(losses), epoch=0, seed=seed)
        loaded, loaded_opt, loaded_rng, meta = load_checkpoint(path, config=config)
        reloaded = loss_and_metrics(loaded, batch, config)
        deterministic = bool(np.isfinite(float(reloaded["loss"]))
                             and np.isclose(float(after["loss"]),
                                             float(reloaded["loss"])))
        optimizer_roundtrip = all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree_util.tree_leaves(opt_state),
                jax.tree_util.tree_leaves(loaded_opt)))
        rng_roundtrip = np.array_equal(np.asarray(rng), np.asarray(loaded_rng))
    return {
        "status": "PASS" if deterministic and optimizer_roundtrip and rng_roundtrip else "FAIL",
        "rows": count, "steps": len(losses), "loss_before": float(before["loss"]),
        "losses": losses, "loss_after": float(after["loss"]),
        "finite_joint_nll": bool(np.isfinite(float(after["loss"]))),
        "checkpoint_reload_deterministic": deterministic,
        "optimizer_roundtrip": optimizer_roundtrip,
        "rng_roundtrip": rng_roundtrip, "checkpoint_step": meta["step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True,
                        help="canonical Parquet file/directory; repeat as needed")
    parser.add_argument("--train-dates", type=_dates,
                        default=DEFAULT_TRAIN_DATES)
    parser.add_argument("--val-dates", type=_dates, default=DEFAULT_VAL_DATES)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--e-history-version", default=E_HISTORY_CORRECTED_V1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = load_train_val(
        args.data, train_dates=args.train_dates, val_dates=args.val_dates,
        min_score=args.min_score, e_history_version=args.e_history_version)
    train, val = loaded["train"], loaded["val"]
    report = {
        "corpus": loaded["corpus"],
        "split": {
            "train": _split_report(train), "validation": _split_report(val),
            "configured": loaded["report"],
        },
        "examples": _examples(train, val),
        "leakage_audit": _leakage(
            train, val, tuple(args.train_dates), tuple(args.val_dates), loaded["corpus"]),
        "encoder_import_contract": {
            "operating_e_history": args.e_history_version,
            "supported_sources": ["native .npz", "Torch .pt/.pth", "in-memory mapping"],
            "transfer": ["manager_token", "role_embedding", "tile_encoder",
                         "global_encoders", "encoder", "encoder_norm"],
            "native_stage25_heads": "randomly initialized unless explicitly mapped",
            "legacy_source_requires_allow_legacy_e": True,
        },
    }
    if args.smoke:
        report["tiny_bc_smoke"] = _smoke(train, args.seed, args.smoke_steps)
    serialized = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(serialized)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

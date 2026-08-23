"""Focused tests for scripts/benchmark_engine_throughput.py.

Covers the measurement-only contracts: deterministic trace generation,
encoding equivalence between the scalar dict path and the native row path,
warmup-aware statistics with loud rejection of impossible numbers, and a
tiny end-to-end fast-engine worker smoke. The official-engine smoke skips
itself without the pinned kaggle-environments install.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_engine_throughput as bench  # noqa: E402


def test_traces_are_pure_functions_of_call_index() -> None:
    assert bench.pass_pair(0) == bench.pass_pair(100)
    assert bench.mixed_pair(3) == bench.mixed_pair(3)
    assert bench.mixed_pair(3) != bench.mixed_pair(4)
    # Deterministic movement cycle and market cadence.
    assert bench.mixed_pair(0)[0]["farmer"] == ["PASS"]
    assert bench.mixed_pair(1)[0]["farmer"] == ["NORTH"]
    assert bench.mixed_pair(1)[0]["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert bench.mixed_pair(3)[0]["market"] == [["SELL", "WHEAT", 1]]


def test_pair_to_rows_matches_scalar_encoding() -> None:
    from fast_env.api import _encode_actions

    for call_index in range(24):
        pair = bench.mixed_pair(call_index)
        rows = bench.pair_to_rows(pair)
        encoded = _encode_actions(list(pair))
        # Farmer slot 0 must match exactly for both seats.
        for seat in range(2):
            assert int(encoded[0, seat, 0, 0]) == rows[seat][0][0]
            assert int(encoded[0, seat, 0, 1]) == rows[seat][0][1]
            assert int(encoded[0, seat, 0, 2]) == rows[seat][0][2]
            # Market orders land after MAX_HANDS+1 slots in the scalar path;
            # the row list appends them contiguously, so compare per order.
            # Market orders land after MAX_HANDS+1 slots in the scalar path;
            # the row list appends them contiguously, so compare per order.
            scalar_market = encoded[0, seat, 241:, :]
            for order_index, row in enumerate(rows[seat][1:]):
                assert list(scalar_market[order_index]) == row


def test_scripted_batch_actions_deterministic_and_shaped() -> None:
    first = bench.scripted_batch_actions(4, 27)
    second = bench.scripted_batch_actions(4, 27)
    assert first.dtype == np.int64
    assert first.shape == (4, 2, 27, 3)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, bench.scripted_batch_actions(4, 27, seed=999))


def test_summarize_discards_warmup_and_validates() -> None:
    stats = bench.summarize([10.0, 1.0, 3.0])
    assert stats["reps_kept"] == 2
    assert stats["median_s"] == 2.0
    assert stats["min_s"] == 1.0
    assert stats["max_s"] == 3.0
    with pytest.raises(ValueError):
        bench.validate_stats({"median_s": float("nan"), "min_s": 1.0, "max_s": 1.0}, "x")
    with pytest.raises(ValueError):
        bench.validate_stats({"median_s": -1.0, "min_s": -2.0, "max_s": 0.5}, "x")
    with pytest.raises(ValueError):
        bench.validate_stats({"median_s": 2.0, "min_s": 3.0, "max_s": 1.0}, "x")


def test_rate_rejects_zero_seconds() -> None:
    assert math.isnan(bench.rate(10.0, 0.0))
    assert bench.rate(10.0, 2.0) == 5.0


def test_worker_smoke_fast_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(
        bench.SUITE_PRESETS,
        "smoke",
        {"scalar_reps": 1, "batch_steps": 2, "batch_reps": 1, "max_batch": 16},
    )
    out = tmp_path / "fast.json"
    # Inline tiny run instead of the preset-driven one so the hardcoded
    # N=512 phase split stays out of the smoke path.
    result = {
        "scalar_episodes": bench.bench_scalar_episodes("fast", 1),
        "batch_throughput": bench.bench_batch("fast", [1, 16], [1], 2, 1),
    }
    out.write_text(json.dumps(result), encoding="utf-8")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scalar_episodes"]["fast_api:mixed"]["step_calls_per_episode"] == 719
    assert data["scalar_episodes"]["fast_native:mixed"]["turns_per_sec_median"] > 0
    for stats in data["batch_throughput"].values():
        assert stats["transitions_per_sec_median"] > 0


def test_official_worker_sections_skip_native_paths(tmp_path, monkeypatch) -> None:
    pytest.importorskip("kaggle_environments")
    memory = bench.bench_memory("official", [128])
    assert set(memory["samples"]) == {"1"}
    profile = bench.bench_profile_scalar("official")
    assert "top_cumulative" in profile
    with pytest.raises(ValueError):
        bench.load_native("official")


def test_render_report_validates_and_produces_markdown(tmp_path) -> None:
    def fake_engine(turns_per_sec: float) -> dict:
        return {
            "environment": {"cpu": "test", "logical_processors": 2},
            "cold_import": {"median_s": 0.1},
            "reset_latency": {"reset": {"median_s": 0.00001}},
            "scalar_episodes": {
                "official:mixed": {
                    "median_s": 1.0, "min_s": 1.0, "max_s": 1.0,
                    "turns_per_sec_median": turns_per_sec,
                    "episodes_per_sec_median": 1.0,
                    "step_calls_per_episode": 719,
                },
            },
            "batch_throughput": {},
            "phase_split_N512": None,
            "memory": {"theoretical_per_env": {}, "samples": {}},
            "profile_scalar": {},
            "profile_batch_N512": None,
        }

    good = {"engines": {"official": fake_engine(500.0)}}
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(good), encoding="utf-8")
    report_path = tmp_path / "report.md"
    bench.main(["report", "--results", str(results_path), "--out", str(report_path)])
    markdown = report_path.read_text(encoding="utf-8")
    assert "# Issue #2 throughput benchmarks" in markdown
    assert "719" in markdown

    bad = {"engines": {"official": fake_engine(float("nan"))}}
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        bench.main(["report", "--results", str(bad_path), "--out", str(tmp_path / "bad.md")])


def test_render_report_scalar_speedup_uses_mixed_rows() -> None:
    def stats(median_s: float) -> dict:
        return {
            "median_s": median_s, "min_s": median_s, "max_s": median_s,
            "turns_per_sec_median": 1.0 / median_s,
            "episodes_per_sec_median": 1.0 / median_s,
            "step_calls_per_episode": 719,
        }

    # pass_only medians would give 4.0x; mixed medians must give ~4.7x.
    def engine(scalar_episodes: dict) -> dict:
        return {
            "environment": {"cpu": "test"},
            "scalar_episodes": scalar_episodes,
            "batch_throughput": {},
            "phase_split_N512": None,
            "memory": {"theoretical_per_env": {}, "samples": {}},
            "profile_scalar": {},
        }

    results = {
        "engines": {
            "official": engine({
                "official:pass_only": stats(2.0),
                "official:mixed": stats(1.3024320999975316),
            }),
            "fast": engine({
                "fast_api:pass_only": stats(0.5),
                "fast_api:mixed": stats(0.2790222499752417),
                "fast_native:mixed": stats(0.0038223500014282763),
            }),
        },
    }
    markdown = bench.render_report(results)
    assert ("**Scalar speedup vs official (mixed trace, full API incl. dict decode): "
            "4.7x.**") in markdown
    assert ":mixed" in markdown
    assert ":pass_only" in markdown

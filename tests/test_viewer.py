"""Focused static/view-model and HTTP smoke tests for the local viewer."""

from __future__ import annotations

import json
import socket
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"


def _write_smoke_trace(path: Path) -> None:
    def farm():
        return {
            "money": 1000,
            "tiles": [[None for _ in range(10)] for _ in range(10)],
            "farmer": [2, 3],
            "hands": [[4, 5]],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }

    def state(step):
        return {
            "step": step,
            "day": 0,
            "hour": step,
            "farms": [farm(), farm()],
            "privates": [
                {"shed": {}, "seeds": {}, "inventories": [{}, {}]},
                {"shed": {}, "seeds": {}, "inventories": [{}, {}]},
            ],
            "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []},
            "rewards": [0, 0],
            "statuses": ["ACTIVE", "ACTIVE"],
        }

    document = {
        "schema_version": 1,
        "metadata": {"seed": 17, "seat": 0, "view": "joint"},
        "turns": [
            {"step": step, "day": 0, "hour": step, "canonical_state": state(step)}
            for step in range(3)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture
def smoke_trace():
    path = ROOT / "artifacts" / "debug_traces" / "_viewer_test_trace.json"
    _write_smoke_trace(path)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_viewer_assets_have_required_controls_and_panels():
    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "trace-file", "drop-zone", "step-slider", "step-back", "step-forward",
        "play-toggle", "speed-select", "seat-toggle", "board", "workers",
        "time-panel", "economy-panel", "storage-panel", "market-panel",
        "manager-panel", "executor-panel", "survival-panel", "actions-panel",
        "trail-toggle", "assignment-toggle", "task-toggle", "urgency-toggle",
        "labels-toggle", "trail-window", "overlay-svg", "overlay-legend",
    ):
        assert f'id="{element_id}"' in html
    assert "styles.css" in html and "viewer.js" in html
    assert "repeat(10" in (VIEWER / "styles.css").read_text(encoding="utf-8")
    assert 'min="8" max="24"' in html
    for semantic in ("feed-survival", "water-must", "water-yield", "harvest", "manager", "blocked", "neutral"):
        assert semantic in html or semantic in (VIEWER / "styles.css").read_text(encoding="utf-8")


def _run_probe(trace: Path | None = None) -> dict:
    command = ["node", str(ROOT / "tests" / "viewer_probe.js")]
    if trace is not None:
        command.append(str(trace))
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def test_viewer_helpers_render_representative_canonical_data_without_mutation():
    result = _run_probe()
    assert result == {"turns": 3, "cells": 100, "workers": 2, "crop": "WHEAT", "animal": "SHEEP", "sidecar": True, "trails": 2}


def test_smoke_trace_loads_through_viewer_helper(smoke_trace):
    result = _run_probe(smoke_trace)
    assert result["turns"] == 3
    assert result["cells"] == 100
    assert result["sidecar"] is False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_viewer_server_serves_only_allowed_assets_and_trace(smoke_trace):
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "viewer.server", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 5
        while True:
            try:
                with urlopen(base + "/viewer/", timeout=1) as response:
                    assert response.status == 200
                break
            except (URLError, TimeoutError, ConnectionError):
                if time.time() >= deadline:
                    raise
                time.sleep(0.05)
        for path in (
            "/viewer/viewer.js",
            "/viewer/styles.css",
            "/artifacts/debug_traces/_viewer_test_trace.json",
        ):
            with urlopen(base + path, timeout=2) as response:
                assert response.status == 200, path
        for path in (
            "/README.md",
            "/rl_manager/cli.py",
            "/viewer/../README.md",
            "/artifacts/debug_traces/../README.md",
            "/artifacts/debug_traces/_viewer_test_trace.txt",
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path, timeout=2)
            assert error.value.code == 404, path
    finally:
        process.terminate()
        process.wait(timeout=5)

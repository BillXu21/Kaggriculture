"""Focused static/view-model and HTTP smoke tests for the local viewer."""

from __future__ import annotations

import json
import socket
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
SMOKE_TRACE = ROOT / "artifacts" / "debug_traces" / "seed_17_seat_0.json"


def test_viewer_assets_have_required_controls_and_panels():
    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "trace-file", "drop-zone", "step-slider", "step-back", "step-forward",
        "play-toggle", "speed-select", "seat-toggle", "board", "workers",
        "time-panel", "economy-panel", "storage-panel", "market-panel",
        "manager-panel", "executor-panel", "survival-panel", "actions-panel",
    ):
        assert f'id="{element_id}"' in html
    assert "styles.css" in html and "viewer.js" in html
    assert "repeat(10" in (VIEWER / "styles.css").read_text(encoding="utf-8")


def _run_probe(trace: Path | None = None) -> dict:
    command = ["node", str(ROOT / "tests" / "viewer_probe.js")]
    if trace is not None:
        command.append(str(trace))
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def test_viewer_helpers_render_representative_canonical_data_without_mutation():
    result = _run_probe()
    assert result == {"turns": 1, "cells": 100, "workers": 2, "crop": "WHEAT", "animal": "SHEEP", "sidecar": True}


def test_smoke_trace_loads_through_viewer_helper():
    assert SMOKE_TRACE.is_file()
    result = _run_probe(SMOKE_TRACE)
    assert result["turns"] == 3
    assert result["cells"] == 100
    assert result["sidecar"] is False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_viewer_server_serves_assets_and_trace():
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
        for path in ("/viewer/viewer.js", "/viewer/styles.css", "/artifacts/debug_traces/seed_17_seat_0.json"):
            with urlopen(base + path, timeout=2) as response:
                assert response.status == 200, path
    finally:
        process.terminate()
        process.wait(timeout=5)

"""Subprocess controller for a local Kaggriculture submission bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any, Mapping


class ExternalAgentError(RuntimeError):
    """An external child failed to load or answer a controller call."""

    def __init__(self, message: str, detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.detail = dict(detail or {})


def bundle_digest(path: str | Path) -> str:
    """Hash one file or a directory's sorted relative file stream."""
    root = Path(path).resolve()
    if root.is_file():
        return _sha256_stream(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for item in sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in {".pyc", ".pyo"}
    ):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entrypoint(source: Path, selected: str) -> tuple[Path, str, str, Path]:
    if ":" in selected:
        target, callable_name = selected.rsplit(":", 1)
    else:
        target, callable_name = selected, "agent"
    if source.is_file():
        bundle = source.parent
        entry = source
        if ":" in selected:
            entry = source if target == "agent.py" else bundle / target
    else:
        bundle = source
        entry = bundle / target
    if entry.suffix == ".py" or entry.is_file():
        return str(entry), "file", callable_name, bundle
    return target, "module", callable_name, bundle


class ExternalController:
    observation_mode = "raw"

    def __init__(
        self,
        *,
        source: str | Path,
        entrypoint: str,
        timeout_seconds: float | None,
    ) -> None:
        self._configuration: dict[str, Any] = {}
        source_path = Path(source).resolve()
        entry, mode, callable_name, bundle = _entrypoint(source_path, entrypoint)
        self._process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("external_agent_child.py")),
                "--entry",
                entry,
                "--mode",
                mode,
                "--callable",
                callable_name,
            ],
            cwd=str(bundle),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._responses: queue.Queue[Any] = queue.Queue()
        self._stderr: list[str] = []
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()
        self.timeout_seconds = timeout_seconds
        ready = self._next("startup")
        if not ready.get("ready"):
            self.close()
            raise ExternalAgentError(
                "external agent failed to load", ready.get("error")
            )

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError as exc:
                self._responses.put(
                    {
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
        self._responses.put(None)

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip("\n"))

    def _next(self, operation: str) -> dict[str, Any]:
        try:
            response = self._responses.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._process.terminate()
            raise ExternalAgentError(
                f"external agent {operation} timed out",
                {
                    "type": "TimeoutError",
                    "timeout_seconds": self.timeout_seconds,
                },
            ) from exc
        if response is None:
            raise ExternalAgentError(
                f"external agent exited during {operation}",
                {
                    "returncode": self._process.poll(),
                    "stderr": list(self._stderr),
                },
            )
        return response

    def act(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._process.poll() is not None:
            raise ExternalAgentError("external agent is no longer running")
        request = {
            "op": "act",
            "observation": observation,
            "configuration": self._configuration,
        }
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        response = self._next("act")
        if not response.get("ok"):
            detail = response.get("error") or {}
            raise ExternalAgentError(
                f"external agent call failed: {detail.get('message', 'unknown error')}",
                detail,
            )
        return response["action"]

    __call__ = act

    def set_configuration(self, configuration: Mapping[str, Any]) -> None:
        self._configuration = dict(configuration)

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.write(json.dumps({"op": "close"}) + "\n")
                    self._process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()


@dataclass
class ExternalControllerFactory:
    source: str | Path
    entrypoint: str = "agent.py:agent"
    timeout_seconds: float | None = None
    display_name: str = "external-agent"

    def __post_init__(self) -> None:
        source = Path(self.source).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")

    @property
    def provenance(self) -> Mapping[str, Any]:
        source = Path(self.source).resolve()
        entry, mode, callable_name, bundle = _entrypoint(source, self.entrypoint)
        return {
            "display_name": self.display_name,
            "kind": "external",
            "identity": f"{self.display_name}:{bundle_digest(source)[:16]}",
            "source_path": str(source),
            "bundle_path": str(bundle),
            "sha256": bundle_digest(source),
            "entrypoint": {
                "selected": self.entrypoint,
                "target": entry,
                "callable": callable_name,
                "mode": mode,
            },
            "callable_contract": "agent(observation) or agent(observation, configuration)",
            "execution_mode": "subprocess",
            "timeout_seconds": self.timeout_seconds,
        }

    def create(
        self,
        *,
        seat: int,
        configuration: Mapping[str, Any],
    ) -> ExternalController:
        del seat
        child = ExternalController(
            source=self.source,
            entrypoint=self.entrypoint,
            timeout_seconds=self.timeout_seconds,
        )
        child.set_configuration(configuration)
        return child


__all__ = [
    "ExternalAgentError",
    "ExternalController",
    "ExternalControllerFactory",
    "bundle_digest",
]

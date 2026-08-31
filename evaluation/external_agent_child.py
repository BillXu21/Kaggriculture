"""Stdlib-only JSONL worker for one downloaded Kaggriculture agent.

This file deliberately has no repository imports.  It is launched in a child
process so target submission modules never enter the evaluator/JAX process.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import traceback


class Struct(dict):
    """Small JSON-compatible equivalent of Kaggle's structified dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _structify(value):
    if isinstance(value, dict):
        return Struct({key: _structify(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_structify(item) for item in value]
    return value


def _load(entry: str, mode: str, callable_name: str):
    if mode == "file":
        path = Path(entry).resolve()
        spec = importlib.util.spec_from_file_location(
            "kaggriculture_external_agent", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load agent file {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(entry)
    try:
        target = getattr(module, callable_name)
    except AttributeError as exc:
        raise AttributeError(
            f"entrypoint {callable_name!r} not found in {entry!r}"
        ) from exc
    if not callable(target):
        raise TypeError(f"entrypoint {callable_name!r} is not callable")
    return target


def _call(target, observation, configuration):
    code = getattr(target, "__code__", None)
    argument_count = code.co_argcount if code is not None else 2
    return target(*[observation, configuration][:argument_count])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True)
    parser.add_argument("--mode", choices=("file", "module"), required=True)
    parser.add_argument("--callable", dest="callable_name", required=True)
    args = parser.parse_args()
    bundle = Path.cwd()
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))
    try:
        # Submission agents occasionally print at import or call time. Keep
        # stdout reserved for the deterministic protocol.
        with contextlib.redirect_stdout(io.StringIO()):
            target = _load(args.entry, args.mode, args.callable_name)
        sys.stdout.write(json.dumps({"ready": True}, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception as exc:  # noqa: BLE001 - serialized child boundary
        sys.stdout.write(
            json.dumps(
                {
                    "ready": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "close":
                break
            if request.get("op") != "act":
                raise ValueError("unknown external-agent operation")
            observation = _structify(request["observation"])
            configuration = _structify(request["configuration"])
            with contextlib.redirect_stdout(io.StringIO()):
                action = _call(target, observation, configuration)
            # Round-trip now, inside the child, to ensure the response is a
            # plain JSON value before it crosses the process boundary.
            action = json.loads(json.dumps(action, allow_nan=False))
            response = {"ok": True, "action": action}
        except Exception as exc:  # noqa: BLE001 - report and keep worker alive
            response = {
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

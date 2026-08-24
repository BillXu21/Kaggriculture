"""Provenance helpers: opening/backend/engine/policy identity capture.

Every episode records exact engine provenance, opening name + digest, and
policy/opponent snapshot identities so any stored trajectory can be tied to
the exact stack that produced it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    """Deterministic JSON string used for all digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def opening_provenance(opening_name: str) -> dict[str, Any]:
    """Built-in opening identity: name plus sha256 of its canonical trace."""
    from opening_book.trace import load_built_in_trace

    trace = load_built_in_trace(opening_name)
    provenance = dict(trace.get("provenance") or {})
    return {
        "name": str(trace.get("identity", opening_name)),
        "digest": sha256_hex(canonical_json(trace)),
        "source_provenance": {
            "source_episode": provenance.get("source_episode"),
            "source_seed": provenance.get("source_seed"),
            "source_player": provenance.get("source_player"),
            "source_seat": provenance.get("source_seat"),
            "source_replay_sha256": provenance.get("source_replay_sha256"),
        },
    }


def backend_provenance(
    backend_name: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact engine provenance for one backend choice/configuration."""
    engine_module = None
    if backend_name == "fast":
        module = sys.modules.get("fast_env._kaggriculture_env")
        engine_module = getattr(module, "__file__", None)
    return {
        "backend": backend_name,
        "configuration": dict(configuration),
        "engine_module": engine_module,
    }

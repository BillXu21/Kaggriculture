"""Official 1.32.7 provenance guard for the differential oracle.

The official engine is the only correctness oracle. This module refuses to run
oracle code against any installed ``kaggle_environments`` that is not exactly
the pinned PyPI wheel content at the pinned upstream commit.

Pin provenance (recorded 2026-08-23):
- PyPI wheel ``kaggle_environments-1.32.7-py3-none-any.whl``
  SHA256 = 2a1bb862ad2d6463080f80f6a766f46d94b53fd57168cfeddb9857fc3dbc4c8f
- Upstream merge commit ``28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c``
- The interpreter files below were verified byte-identical between the
  installed wheel and ``raw.githubusercontent.com`` at that commit.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from typing import Any, Callable

OFFICIAL_PACKAGE_NAME = "kaggle-environments"
OFFICIAL_PACKAGE_VERSION = "1.32.7"
OFFICIAL_UPSTREAM_COMMIT = "28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c"
OFFICIAL_WHEEL_SHA256 = "2a1bb862ad2d6463080f80f6a766f46d94b53fd57168cfeddb9857fc3dbc4c8f"

# Relative to kaggle_environments/envs/kaggriculture/.
OFFICIAL_FILE_SHA256 = {
    "kaggriculture.json": "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867",
    "kaggriculture.py": "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
}


class ProvenanceError(RuntimeError):
    """Raised when the installed official engine does not match the pin."""


def _package_version(kaggle_environments: Any) -> str:
    from importlib.metadata import version as dist_version

    return dist_version(OFFICIAL_PACKAGE_NAME)


def verify_official_provenance() -> dict[str, Any]:
    """Verify installed version and interpreter file hashes against the pin.

    Returns a provenance report dict; raises :class:`ProvenanceError` on any
    mismatch. Imports ``kaggle_environments`` lazily so normal fast-engine use
    never pays for it.
    """
    try:
        kaggle_environments = importlib.import_module("kaggle_environments")
    except ImportError as error:
        raise ProvenanceError(
            "kaggle_environments is not installed; install the pinned "
            f"{OFFICIAL_PACKAGE_NAME}=={OFFICIAL_PACKAGE_VERSION} wheel to use the oracle"
        ) from error

    report: dict[str, Any] = {
        "package": OFFICIAL_PACKAGE_NAME,
        "expected_version": OFFICIAL_PACKAGE_VERSION,
        "upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
        "wheel_sha256": OFFICIAL_WHEEL_SHA256,
        "files": {},
    }

    installed_version = _package_version(kaggle_environments)
    report["installed_version"] = installed_version
    if installed_version != OFFICIAL_PACKAGE_VERSION:
        raise ProvenanceError(
            f"installed kaggle_environments version {installed_version!r} != pinned "
            f"{OFFICIAL_PACKAGE_VERSION!r}; refusing to run the oracle"
        )

    env_dir = os.path.join(os.path.dirname(kaggle_environments.__file__), "envs", "kaggriculture")
    for relative_name, expected_hash in sorted(OFFICIAL_FILE_SHA256.items()):
        file_path = os.path.join(env_dir, relative_name)
        if not os.path.isfile(file_path):
            raise ProvenanceError(f"missing official interpreter file: {file_path}")
        digest = hashlib.sha256(open(file_path, "rb").read()).hexdigest()
        report["files"][relative_name] = digest
        if digest != expected_hash:
            raise ProvenanceError(
                f"{relative_name} sha256 {digest} != pinned {expected_hash} "
                f"(upstream commit {OFFICIAL_UPSTREAM_COMMIT}); refusing to run the oracle"
            )
    return report


def require_official_modules() -> tuple[Any, ...]:
    """Lazily import and return (make, ProvenanceError-free) official modules.

    Only oracle/evaluation code may call this. Runs the provenance guard first.
    """
    verify_official_provenance()
    core = importlib.import_module("kaggle_environments")
    make: Callable[..., Any] = core.make
    return make

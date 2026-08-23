"""Fresh-process import isolation: the fast hot path must stay Kaggle-free."""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_fast_env_import_in_fresh_process_stays_kaggle_free() -> None:
    result = _run(
        "import sys\n"
        "import fast_env\n"
        "env = fast_env.FastKaggricultureEnv({'seed': 7})\n"
        "env.reset()\n"
        "env.step([{'farmer': ['PASS'], 'hands': [], 'market': []}] * 2)\n"
        "assert 'kaggle_environments' not in sys.modules, 'kaggle leaked'\n"
        "assert 'open_spiel' not in sys.modules, 'open_spiel leaked'\n"
        "print('isolated')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "isolated" in result.stdout


def test_oracle_package_import_is_lazy_about_kaggle() -> None:
    result = _run(
        "import sys\n"
        "import oracle\n"
        "assert 'kaggle_environments' not in sys.modules, "
        "'oracle import eagerly loaded kaggle_environments'\n"
        "print('lazy')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "lazy" in result.stdout

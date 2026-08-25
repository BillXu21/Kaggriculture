from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_pure_fast_env_imports_without_native_extension() -> None:
    result = _run(
        "import sys\n"
        "sys.modules['fast_env._kaggriculture_env'] = None\n"
        "import fast_env\n"
        "import fast_env.market\n"
        "from fast_env import HINGE_GAIN, market_price\n"
        "assert HINGE_GAIN == 8.0\n"
        "assert market_price('CARROT', 9_550) == 70\n"
        "assert market_price('TOMATO', 9_800) == 84\n"
        "assert fast_env.__all__ == ['FastKaggricultureEnv', 'HINGE_GAIN', 'market_price']\n"
        "assert {'FastKaggricultureEnv', 'HINGE_GAIN', 'market_price'} <= set(dir(fast_env))\n"
        "assert 'fast_env.api' not in sys.modules\n"
        "try:\n"
        "    fast_env.FastKaggricultureEnv\n"
        "except ModuleNotFoundError as error:\n"
        "    assert error.name == 'fast_env._kaggriculture_env'\n"
        "else:\n"
        "    raise AssertionError('native access unexpectedly succeeded')\n"
        "print('pure-imports-and-lazy-error-ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "pure-imports-and-lazy-error-ok" in result.stdout


def test_fast_env_export_loads_with_compatible_native_module() -> None:
    result = _run(
        "import sys\n"
        "import types\n"
        "native = types.ModuleType('fast_env._kaggriculture_env')\n"
        "native.ACTION_SLOTS = 251\n"
        "native.MAX_HANDS = 240\n"
        "native.RustBatchEnv = type('RustBatchEnv', (), {})\n"
        "sys.modules['fast_env._kaggriculture_env'] = native\n"
        "from fast_env import FastKaggricultureEnv\n"
        "assert FastKaggricultureEnv.__name__ == 'FastKaggricultureEnv'\n"
        "assert FastKaggricultureEnv.__module__ == 'fast_env.api'\n"
        "print('mocked-native-export-ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "mocked-native-export-ok" in result.stdout

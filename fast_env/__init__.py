"""Narrow, framework-independent Kaggriculture 1.32.7 environment API."""

from .market import HINGE_GAIN, market_price

__all__ = ["FastKaggricultureEnv", "HINGE_GAIN", "market_price"]


def __getattr__(name: str):
    if name == "FastKaggricultureEnv":
        from .api import FastKaggricultureEnv

        return FastKaggricultureEnv
    if name == "BatchedFastEnv":
        from .batch import BatchedFastEnv

        return BatchedFastEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .base import BasePriceProvider, PriceProviderError

ProviderName = Literal["yfinance", "stooq", "sample"]


def make_provider(name: str, sample_dir: Path) -> BasePriceProvider:
    """Factory with *lazy imports*.

    This keeps the project runnable even if you don't have every optional provider dependency installed.
    """
    n = (name or "yfinance").strip().lower()
    if n == "yfinance":
        from .yfinance_provider import YFinanceProvider  # lazy import
        return YFinanceProvider()
    if n == "stooq":
        from .stooq_provider import StooqProvider  # lazy import
        return StooqProvider()
    if n == "sample":
        from .sample_provider import SampleCSVProvider  # lazy import
        return SampleCSVProvider(sample_dir)
    raise ValueError(f"Unknown provider: {name}")

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ..providers import make_provider
from ..providers.base import BasePriceProvider, PriceProviderError
from ..config import Settings


def _parse_lookback_to_timedelta(lookback: str) -> Optional[timedelta]:
    lb = (lookback or "").strip().lower()
    if lb in {"", "max"}:
        return None

    try:
        if lb.endswith("y"):
            years = float(lb[:-1])
            return timedelta(days=int(years * 365))
        if lb.endswith("mo"):
            months = float(lb[:-2])
            return timedelta(days=int(months * 30))
        if lb.endswith("d"):
            days = float(lb[:-1])
            return timedelta(days=int(days))
    except Exception:
        return None

    return None


@dataclass
class DataAgent:
    settings: Settings
    provider: BasePriceProvider

    @classmethod
    def from_settings(cls, settings: Settings) -> "DataAgent":
        provider = make_provider(settings.data_provider, settings.sample_data_dir)
        return cls(settings=settings, provider=provider)

    def _cache_path(self, ticker: str, timeframe: str) -> Path:
        safe_ticker = ticker.upper().replace("/", "_")
        return self.settings.cache_dir / f"{self.settings.data_provider}_{safe_ticker}_{timeframe}.csv"

    def get_ohlcv(self, ticker: str, timeframe: str, lookback: str) -> pd.DataFrame:
        """Get OHLCV with caching.

        If provider fails and offline_ok is True, will try sample provider as a fallback if available.
        """
        cache_path = self._cache_path(ticker, timeframe)
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            df = df.rename(columns={c: c.title() for c in df.columns})
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
            else:
                # if index was saved as unnamed
                df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        else:
            try:
                df = self.provider.fetch_ohlcv(ticker=ticker, timeframe=timeframe, lookback=lookback)
            except PriceProviderError as e:
                if not self.settings.offline_ok:
                    raise
                # fallback to sample provider
                fallback = make_provider("sample", self.settings.sample_data_dir)
                df = fallback.fetch_ohlcv(ticker=ticker, timeframe=timeframe, lookback=lookback)
            # Save cache
            tmp = df.copy()
            tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "Date"})
            tmp.to_csv(cache_path, index=False)

        # Apply lookback trim for providers that don't support it well
        delta = _parse_lookback_to_timedelta(lookback)
        if delta is not None and not df.empty:
            end = df.index.max()
            start = end - delta
            df = df[df.index >= start]

        return df

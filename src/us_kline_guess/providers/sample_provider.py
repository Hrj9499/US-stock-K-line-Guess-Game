from __future__ import annotations

from pathlib import Path
import pandas as pd

from .base import BasePriceProvider, PriceProviderError


class SampleCSVProvider(BasePriceProvider):
    def __init__(self, sample_dir: Path):
        self.sample_dir = sample_dir

    def fetch_ohlcv(self, ticker: str, timeframe: str, lookback: str) -> pd.DataFrame:
        if timeframe != "1d":
            raise PriceProviderError("Toy system currently supports timeframe=1d only")

        path = self.sample_dir / f"{ticker.upper()}_{timeframe}.csv"
        if not path.exists():
            raise PriceProviderError(f"sample file not found: {path.name}")

        df = pd.read_csv(path)
        if df.empty:
            raise PriceProviderError(f"sample file empty: {path.name}")

        # Standardize
        df = df.rename(columns={c: c.title() for c in df.columns})
        if "Date" not in df.columns:
            raise PriceProviderError("sample file missing Date column")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        for c in ["Open", "High", "Low", "Close"]:
            if c not in df.columns:
                raise PriceProviderError(f"sample file missing column {c}")

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df

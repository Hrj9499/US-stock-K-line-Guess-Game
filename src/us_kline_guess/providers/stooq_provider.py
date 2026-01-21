from __future__ import annotations

import io
import re
import requests
import pandas as pd

from .base import BasePriceProvider, PriceProviderError


class StooqProvider(BasePriceProvider):
    """Fetches daily OHLCV from Stooq's free CSV endpoint.

    Endpoint format:
      https://stooq.com/q/d/l/?s=aapl.us&i=d

    Notes:
    - This is free, no API key.
    - Symbols are lowercased and usually end with '.us' for US stocks.
    """

    @staticmethod
    def _to_stooq_symbol(ticker: str) -> str:
        t = ticker.strip().lower()
        # common US ticker normalization: BRK-B -> brk.b.us
        t = t.replace("-", ".")
        # remove spaces
        t = re.sub(r"\s+", "", t)
        if not t.endswith(".us"):
            t = f"{t}.us"
        return t

    def fetch_ohlcv(self, ticker: str, timeframe: str, lookback: str) -> pd.DataFrame:
        if timeframe != "1d":
            raise PriceProviderError("Toy system currently supports timeframe=1d only")
        # Stooq doesn't support lookback param in the same way; we fetch full history and trim later.
        symbol = self._to_stooq_symbol(ticker)
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
        except Exception as e:
            raise PriceProviderError(f"stooq request failed: {e}") from e

        text = r.text
        if "Date,Open,High,Low,Close" not in text:
            raise PriceProviderError("stooq returned unexpected content (maybe invalid symbol)")

        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            raise PriceProviderError("stooq returned empty data")

        # Standardize
        df = df.rename(columns={c: c.title() for c in df.columns})
        if "Volume" not in df.columns:
            df["Volume"] = 0

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df

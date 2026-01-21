from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PriceProviderError(RuntimeError):
    pass


class BasePriceProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(self, ticker: str, timeframe: str, lookback: str) -> pd.DataFrame:
        """Return a DataFrame indexed by datetime with columns:
        Open, High, Low, Close, Volume (float/int).
        """
        raise NotImplementedError

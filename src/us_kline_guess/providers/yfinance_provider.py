from __future__ import annotations

import pandas as pd
import yfinance as yf

from .base import BasePriceProvider, PriceProviderError


_OHLCV_KEYS = {"open", "high", "low", "close", "adj close", "volume"}


def _normalize_yfinance_df(df: pd.DataFrame, requested_ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to a single-ticker OHLCV dataframe.

    Why needed:
    - yfinance may return MultiIndex columns (tuples) for certain inputs or versions.
      Our toy system expects plain columns: Open/High/Low/Close/Volume.
    """
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = pd.Index(df.columns.get_level_values(0)).astype(str).str.strip().str.lower()
        lv1 = pd.Index(df.columns.get_level_values(1)).astype(str).str.strip().str.lower()

        lv0_has_ohlc = {"open", "high", "low", "close"}.issubset(set(lv0))
        lv1_has_ohlc = {"open", "high", "low", "close"}.issubset(set(lv1))

        # Case A: columns are (field, ticker)
        if lv0_has_ohlc and not lv1_has_ohlc:
            tickers = list(pd.Index(df.columns.get_level_values(1)).unique().astype(str))
            if len(tickers) > 1:
                chosen = requested_ticker if requested_ticker in tickers else tickers[0]
                df = df.xs(chosen, axis=1, level=1, drop_level=True)
            else:
                df.columns = df.columns.get_level_values(0)

        # Case B: columns are (ticker, field)
        elif lv1_has_ohlc:
            tickers = list(pd.Index(df.columns.get_level_values(0)).unique().astype(str))
            if len(tickers) > 1:
                chosen = requested_ticker if requested_ticker in tickers else tickers[0]
                df = df.xs(chosen, axis=1, level=0, drop_level=True)
            else:
                df.columns = df.columns.get_level_values(1)

        else:
            # Fallback: keep the last element of each tuple
            df.columns = [c[-1] for c in df.columns]

    # Sometimes you can still get tuple-like columns without MultiIndex
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

    # Standardize names
    df = df.rename(columns=lambda x: str(x).strip().title())
    return df


class YFinanceProvider(BasePriceProvider):
    def fetch_ohlcv(self, ticker: str, timeframe: str, lookback: str) -> pd.DataFrame:
        if timeframe != "1d":
            raise PriceProviderError("Toy system currently supports timeframe=1d only")

        ticker = (ticker or "").strip().upper()
        if not ticker:
            raise PriceProviderError("ticker is empty")

        try:
            df = yf.download(
                ticker,
                period=lookback,
                interval=timeframe,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            raise PriceProviderError(f"yfinance download failed: {e}") from e

        if df is None or df.empty:
            raise PriceProviderError("yfinance returned empty data")

        df = _normalize_yfinance_df(df, requested_ticker=ticker)

        required = ["Open", "High", "Low", "Close"]
        for c in required:
            if c not in df.columns:
                raise PriceProviderError(
                    f"missing column {c} from yfinance response. columns={list(df.columns)[:10]}"
                )

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df

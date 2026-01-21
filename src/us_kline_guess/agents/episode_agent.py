from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..models import Candle, EpisodePrivate
from ..config import Settings
from .data_agent import DataAgent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_candles(df: pd.DataFrame) -> list[Candle]:
    out: list[Candle] = []
    for idx, row in df.iterrows():
        out.append(Candle(
            t=idx.strftime("%Y-%m-%d"),
            o=float(row["Open"]),
            h=float(row["High"]),
            l=float(row["Low"]),
            c=float(row["Close"]),
            v=float(row.get("Volume", 0.0)),
        ))
    return out


@dataclass
class EpisodeAgent:
    settings: Settings
    data_agent: DataAgent

    def generate_episode(
        self,
        ticker: str,
        timeframe: str,
        lookback: str,
        bars: int,
        hide_n: int,
        seed: Optional[int] = None,
    ) -> EpisodePrivate:
        if timeframe != "1d":
            raise ValueError("Toy system currently supports timeframe=1d only")

        df = self.data_agent.get_ohlcv(ticker=ticker, timeframe=timeframe, lookback=lookback)
        df = df.dropna()

        need = bars + hide_n
        if len(df) < need + 10:
            raise ValueError(f"Not enough data for {ticker}. Need at least {need+10} rows, got {len(df)}")

        if seed is None:
            seed = int(datetime.now(timezone.utc).timestamp())
        rng = random.Random(seed)

        start_idx = rng.randint(0, len(df) - need)
        seg = df.iloc[start_idx : start_idx + need]
        candles = _to_candles(seg)

        shown = candles[:-hide_n]
        truth = candles[-hide_n:]

        last_shown_close = shown[-1].c
        last_truth_close = truth[-1].c
        truth_direction = "UP" if last_truth_close > last_shown_close else "DOWN"

        episode_id = f"{ticker.upper()}-{seg.index[0].strftime('%Y%m%d')}-{seg.index[-1].strftime('%Y%m%d')}-{seed}"

        question = f"猜未来 {hide_n} 根K线整体方向（UP/DOWN）"
        ep = EpisodePrivate(
            episode_id=episode_id,
            ticker=ticker.upper(),
            timeframe=timeframe,
            bars=bars,
            hide_n=hide_n,
            created_at=_now_iso(),
            start=seg.index[0].strftime("%Y-%m-%d"),
            end=seg.index[-1].strftime("%Y-%m-%d"),
            candles=shown,
            hidden_truth=truth,
            truth_direction=truth_direction,  # type: ignore
            question=question,
            answer_type="direction",
            meta={
                "seed": seed,
                "lookback": lookback,
                "provider": self.settings.data_provider,
                "truth_direction": truth_direction,
                "last_shown_close": float(last_shown_close),
                "last_truth_close": float(last_truth_close),
            },
        )
        return ep

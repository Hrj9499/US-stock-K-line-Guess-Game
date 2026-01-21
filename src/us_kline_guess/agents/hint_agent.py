from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..models import EpisodePrivate, HintResponse
from ..config import Settings
from .data_agent import DataAgent


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    # simple EMA implementation (no pandas dependency in core loop)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    prev_close = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))
    atr_series = _ema(tr, span=period)
    return float(atr_series[-1])


def _segment_vector(closes: np.ndarray) -> np.ndarray:
    # Vectorize by log returns, z-score normalize within the segment
    r = np.diff(np.log(closes.astype(np.float64)))
    if r.size == 0:
        return np.zeros((1,), dtype=np.float32)
    r = (r - r.mean()) / (r.std() + 1e-8)
    v = r.astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


@dataclass
class HintAgent:
    settings: Settings
    data_agent: DataAgent

    def get_hint(self, episode: EpisodePrivate, level: int = 1) -> HintResponse:
        level = int(level)
        if level < 1 or level > 3:
            raise ValueError("level must be 1..3")

        closes = np.array([c.c for c in episode.candles], dtype=np.float64)
        highs = np.array([c.h for c in episode.candles], dtype=np.float64)
        lows = np.array([c.l for c in episode.candles], dtype=np.float64)
        vols = np.array([c.v for c in episode.candles], dtype=np.float64)

        # Basic features
        ret_20 = float(closes[-1] / closes[-21] - 1.0) if len(closes) >= 21 else float("nan")
        ma5 = float(closes[-5:].mean()) if len(closes) >= 5 else float("nan")
        ma20 = float(closes[-20:].mean()) if len(closes) >= 20 else float("nan")
        atr14 = _atr(highs, lows, closes, period=14)
        atr_pct = _safe_div(atr14, closes[-1]) if np.isfinite(atr14) else float("nan")

        # Support/Resistance proxies
        window = min(40, len(closes))
        recent_high = float(highs[-window:].max())
        recent_low = float(lows[-window:].min())
        dist_to_high = float((recent_high - closes[-1]) / closes[-1])
        dist_to_low = float((closes[-1] - recent_low) / closes[-1])

        # Volume change
        vol_ratio = float(vols[-5:].mean() / (vols[-20:].mean() + 1e-8)) if len(vols) >= 20 else float("nan")

        if level == 1:
            trend = "偏强" if (np.isfinite(ma5) and np.isfinite(ma20) and ma5 >= ma20) else "偏弱"
            hint = (
                f"形态概览（不构成投资建议）：\n"
                f"- 最近20根累计涨跌：{ret_20:+.2%}（越大越偏强）\n"
                f"- MA5 vs MA20：MA5={ma5:.2f}，MA20={ma20:.2f} → {trend}\n"
                f"- 波动（ATR14/Close）：{atr_pct:.2%}（越大越“难猜”）"
            )
            stats = {
                "ret_20": ret_20,
                "ma5": ma5,
                "ma20": ma20,
                "atr14": atr14,
                "atr_pct": atr_pct,
            }
            return HintResponse(episode_id=episode.episode_id, level=1, title="轻提示：趋势与波动", hint=hint, stats=stats)

        if level == 2:
            # Identify simple pattern descriptors
            is_range = (dist_to_high < 0.05 and dist_to_low < 0.05) if (np.isfinite(dist_to_high) and np.isfinite(dist_to_low)) else False
            regime = "震荡" if is_range else ("突破前夕/趋势中" if dist_to_high < dist_to_low else "回撤/下行压力中")

            hint = (
                f"结构提示（不构成投资建议）：\n"
                f"- 近{window}根区间：High={recent_high:.2f}，Low={recent_low:.2f}\n"
                f"- 距离区间上沿：{dist_to_high:.2%}；距离区间下沿：{dist_to_low:.2%}\n"
                f"- 量能变化（近5 vs 近20均量）：{vol_ratio:.2f} 倍\n"
                f"- 粗略判断：更像「{regime}」\n"
                f"玩法建议：你可以把‘关键位’当成未来{episode.hide_n}根里更容易被触发的事件（是否靠近/是否突破）。"
            )
            stats = {
                "recent_high": recent_high,
                "recent_low": recent_low,
                "dist_to_high": dist_to_high,
                "dist_to_low": dist_to_low,
                "vol_ratio": vol_ratio,
            }
            return HintResponse(episode_id=episode.episode_id, level=2, title="中提示：区间与关键位", hint=hint, stats=stats)

        # level 3: similarity-based "RAG-ish" hint
        df = self.data_agent.get_ohlcv(
            ticker=episode.ticker,
            timeframe=episode.timeframe,
            lookback=episode.meta.get("lookback", self.settings.default_lookback),
        ).dropna()

        bars = episode.bars
        hide_n = episode.hide_n
        need = bars + hide_n
        if len(df) < need + 5:
            hint = "历史数据不足，无法做相似形态检索。"
            return HintResponse(episode_id=episode.episode_id, level=3, title="重提示：相似形态检索", hint=hint, stats={})

        target_vec = _segment_vector(closes)

        sims: list[tuple[float, int, float]] = []  # (sim, start_idx, future_ret)
        close_series = df["Close"].to_numpy(dtype=np.float64)

        for i in range(0, len(df) - need):
            seg_closes = close_series[i : i + bars]
            vec = _segment_vector(seg_closes)
            sim = float(np.dot(target_vec, vec))
            # future return over hide_n horizon
            shown_end = close_series[i + bars - 1]
            future_end = close_series[i + need - 1]
            future_ret = float(future_end / shown_end - 1.0)
            sims.append((sim, i, future_ret))

        sims.sort(key=lambda x: x[0], reverse=True)
        topk = sims[: max(5, min(self.settings.similar_topk, len(sims)))]
        rets = np.array([r for _, _, r in topk], dtype=np.float64)

        up_rate = float((rets > 0).mean()) if rets.size else float("nan")
        avg_ret = float(rets.mean()) if rets.size else float("nan")
        med_ret = float(np.median(rets)) if rets.size else float("nan")
        q10 = float(np.quantile(rets, 0.10)) if rets.size else float("nan")
        q90 = float(np.quantile(rets, 0.90)) if rets.size else float("nan")

        # include a few examples (start date + sim + future_ret)
        examples: list[dict[str, Any]] = []
        for sim, i, r in topk[:5]:
            start_date = df.index[i].strftime("%Y-%m-%d")
            end_date = df.index[i + bars - 1].strftime("%Y-%m-%d")
            examples.append({"start": start_date, "end_shown": end_date, "sim": sim, "future_ret": r})

        hint = (
            f"相似形态检索（仅作游戏提示，不构成投资建议）：\n"
            f"- 在历史数据中找到 {len(topk)} 段与当前片段最相似的走势（用“收益序列相似度”衡量）。\n"
            f"- 这些相似片段在未来 {hide_n} 根的统计：\n"
            f"  · 上涨概率（future_ret>0）：{up_rate:.0%}\n"
            f"  · 平均回报：{avg_ret:+.2%}；中位数：{med_ret:+.2%}\n"
            f"  · 10%分位：{q10:+.2%}；90%分位：{q90:+.2%}\n"
            f"- 示例（前5个相似片段）：{examples}\n"
            f"玩法建议：把它当成‘历史类比’，不是确定答案。"
        )

        stats = {
            "topk": len(topk),
            "up_rate": up_rate,
            "avg_ret": avg_ret,
            "median_ret": med_ret,
            "q10": q10,
            "q90": q90,
            "examples": examples,
        }
        return HintResponse(episode_id=episode.episode_id, level=3, title="重提示：相似形态检索", hint=hint, stats=stats)

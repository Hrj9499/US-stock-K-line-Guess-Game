from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from us_kline_guess.config import get_settings
from us_kline_guess.agents.data_agent import DataAgent
from us_kline_guess.agents.episode_agent import EpisodeAgent
from us_kline_guess.agents.hint_agent import HintAgent
from us_kline_guess.agents.scoring_agent import ScoringAgent
from us_kline_guess.models import SubmitAnswerRequest
from us_kline_guess.storage.file_store import EpisodeStore, ScoreStore


# ---- Optional LLM (Gemini) imports (graceful fallback) ----
GeminiClient = None
LLMExplainAgent = None
try:
    from us_kline_guess.llm.gemini_client import GeminiClient as _GeminiClient
    from us_kline_guess.agents.llm_explain_agent import LLMExplainAgent as _LLMExplainAgent

    GeminiClient = _GeminiClient
    LLMExplainAgent = _LLMExplainAgent
except Exception:
    GeminiClient = None
    LLMExplainAgent = None


# ------------------------- i18n -------------------------
TEXT = {
    "en": {
        "title": "📈 US Stock K-line Guess Game",
        "game_settings": "Game Settings",
        "language": "Language / 语言",
        "data_provider": "Data provider",
        "ticker": "Ticker",
        "lookback": "Lookback",
        "bars_shown": "Bars shown",
        "hidden_bars": "Hidden bars",
        "hint_level": "Hint level",
        "user_id": "User ID",
        "indicators": "Indicators",
        "ma_periods": "MA periods",
        "rsi_period": "RSI period",
        "show_volume": "Show volume",
        "show_rsi": "Show RSI",
        "llm": "LLM (Gemini)",
        "use_llm": "Use LLM explanation",
        "missing_key": "Missing GOOGLE_API_KEY. LLM enabled but cannot call.",
        "missing_llm_code": "LLM code not found/import failed. Check gemini_client.py & llm_explain_agent.py.",
        "disclaimer": "⚠️ For training/entertainment only. Not financial advice.",
        "new_episode": "🎲 New Episode",
        "next": "⏭️ Next",
        "reset": "🧹 Reset",
        "generating": "Generating episode...",
        "generating_next": "Generating next episode...",
        "start_info": "Click **New Episode** to start. Suggest provider=sample for offline stable run.",
        "episode": "Episode",
        "question": "🎯 Question",
        "snapshot": "📌 Snapshot (last shown bar)",
        "close": "Close",
        "vol": "Volume",
        "vol_ratio": "Vol(5/20)",
        "hint": "💡 Hint",
        "get_hint": "Get Hint",
        "view_hint": "View hint",
        "your_guess": "🎮 Your Guess",
        "up": "🟢 UP",
        "down": "🔴 DOWN",
        "result": "🧾 Result",
        "correct": "✅ Correct!",
        "wrong": "❌ Wrong",
        "your_guess_label": "Your Guess",
        "truth": "Truth",
        "score_delta": "Score Δ",
        "total_score": "Total Score",
        "explanation": "🧠 Explanation (LLM)",
        "llm_unavailable": "LLM not enabled or unavailable. Check UKG_USE_LLM=1, GOOGLE_API_KEY, and llm files/deps.",
        "view_expl": "📌 View explanation",
        "debug": "Show raw details (debug)",
        "leaderboard": "🏆 Leaderboard (Top 10)",
        "created": "Created",
        "shown_title": "Shown candles + indicators",
        "revealed_title": "Revealed (shown + hidden) + indicators",
    },
    "zh": {
        "title": "📈 美股 K 线猜猜乐",
        "game_settings": "游戏设置",
        "language": "Language / 语言",
        "data_provider": "数据源",
        "ticker": "股票代码",
        "lookback": "回看区间",
        "bars_shown": "显示K线数量",
        "hidden_bars": "隐藏K线数量",
        "hint_level": "提示等级",
        "user_id": "用户ID",
        "indicators": "指标",
        "ma_periods": "均线周期",
        "rsi_period": "RSI周期",
        "show_volume": "显示成交量",
        "show_rsi": "显示RSI",
        "llm": "LLM（Gemini）",
        "use_llm": "启用LLM解释",
        "missing_key": "缺少 GOOGLE_API_KEY。LLM 已启用但无法调用。",
        "missing_llm_code": "LLM 代码未找到/导入失败。检查 gemini_client.py 与 llm_explain_agent.py。",
        "disclaimer": "⚠️ 仅用于训练/娱乐，不构成投资建议。",
        "new_episode": "🎲 生成新题",
        "next": "⏭️ 下一题",
        "reset": "🧹 重置",
        "generating": "正在生成题目...",
        "generating_next": "正在生成下一题...",
        "start_info": "点击 **生成新题** 开始。建议先用 provider=sample（离线稳定）。",
        "episode": "关卡",
        "question": "🎯 题目",
        "snapshot": "📌 快照（最后一根已显示K线）",
        "close": "收盘价",
        "vol": "成交量",
        "vol_ratio": "量比(5/20)",
        "hint": "💡 提示",
        "get_hint": "获取提示",
        "view_hint": "查看提示",
        "your_guess": "🎮 你的选择",
        "up": "🟢 上涨 UP",
        "down": "🔴 下跌 DOWN",
        "result": "🧾 结果",
        "correct": "✅ 正确！",
        "wrong": "❌ 错误",
        "your_guess_label": "你的答案",
        "truth": "正确答案",
        "score_delta": "本题得分",
        "total_score": "总分",
        "explanation": "🧠 复盘解释（LLM）",
        "llm_unavailable": "LLM 未启用或不可用（检查：UKG_USE_LLM=1、GOOGLE_API_KEY、以及 llm 文件/依赖）。",
        "view_expl": "📌 查看复盘解释",
        "debug": "显示原始详情（debug）",
        "leaderboard": "🏆 排行榜（前10）",
        "created": "已生成",
        "shown_title": "已显示K线 + 指标",
        "revealed_title": "揭晓（已显示 + 隐藏）+ 指标",
    },
}


def tr(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, key)


# ------------------------- Indicator helpers -------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI implemented via EWMA."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame, ma_periods: Iterable[int], rsi_period: int) -> pd.DataFrame:
    """
    Fix 1: MA line missing in early part:
      rolling mean produces NaN for first (window-1) points.
      For UI clarity we set min_periods=1 so MA starts from the first candle.
    """
    out = df.copy()
    close = out["c"]

    for p in sorted(set(int(x) for x in ma_periods)):
        if p <= 1:
            continue
        out[f"ma{p}"] = close.rolling(window=p, min_periods=1).mean()

    out[f"rsi{rsi_period}"] = compute_rsi(close, period=rsi_period)
    return out


def candles_to_df(candles) -> pd.DataFrame:
    df = pd.DataFrame([c.model_dump() for c in candles])
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t")
    return df


def plot_kline_with_indicators(
    df: pd.DataFrame,
    title: str,
    ma_periods: list[int],
    rsi_period: int,
    show_volume: bool = True,
    show_rsi: bool = True,
) -> go.Figure:
    """
    Fix 2: title/legend overlap.
    - Hide candlestick legend (it takes space and isn't that useful)
    - Move legend above plot with enough top margin
    - Place title slightly higher and left aligned
    """
    rows = 1 + int(show_volume) + int(show_rsi)
    row_heights = [0.62, 0.20, 0.18] if rows == 3 else ([0.75, 0.25] if rows == 2 else [1.0])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Row 1: Candlestick + MA
    fig.add_trace(
        go.Candlestick(
            x=df["t"],
            open=df["o"],
            high=df["h"],
            low=df["l"],
            close=df["c"],
            name="K",
            showlegend=False,  # reduce legend clutter
        ),
        row=1,
        col=1,
    )

    for p in ma_periods:
        col = f"ma{p}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["t"], y=df[col], mode="lines", name=f"MA{p}"),
                row=1,
                col=1,
            )

    cur_row = 1

    # Volume row
    if show_volume:
        cur_row += 1
        fig.add_trace(go.Bar(x=df["t"], y=df["v"], name="Volume", opacity=0.55), row=cur_row, col=1)

    # RSI row
    if show_rsi:
        cur_row += 1
        rsi_col = f"rsi{rsi_period}"
        if rsi_col in df.columns:
            fig.add_trace(go.Scatter(x=df["t"], y=df[rsi_col], mode="lines", name=f"RSI{rsi_period}"), row=cur_row, col=1)
            try:
                fig.add_hline(y=70, line_dash="dash", row=cur_row, col=1)
                fig.add_hline(y=30, line_dash="dash", row=cur_row, col=1)
            except Exception:
                pass
            fig.update_yaxes(range=[0, 100], row=cur_row, col=1)

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top"),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        height=700,
        # Bigger top margin so legend won't collide with title
        margin=dict(l=10, r=10, t=110, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,   # move legend above plot area
            xanchor="left",
            x=0,
        ),
    )
    return fig


def format_leaderboard(top: list[tuple[str, int]], current_user: str | None, lang: str) -> pd.DataFrame:
    if lang == "zh":
        df = pd.DataFrame(top, columns=["用户", "分数"])
        df.insert(0, "排名", range(1, len(df) + 1))
        if current_user:
            df.insert(1, "我", df["用户"].apply(lambda x: "👤" if x == current_user else ""))
        return df
    else:
        df = pd.DataFrame(top, columns=["user", "score"])
        df.insert(0, "rank", range(1, len(df) + 1))
        if current_user:
            df.insert(1, "you", df["user"].apply(lambda x: "👤" if x == current_user else ""))
        return df


def reset_round_state() -> None:
    st.session_state.last_hint = None
    st.session_state.last_hint_key = None
    st.session_state.last_result = None
    st.session_state.last_explanation = None
    st.session_state.last_expl_key = None


# ------------------------- Page setup -------------------------
default_lang = os.getenv("UKG_LANG", "en").lower()
default_lang = "zh" if default_lang.startswith("zh") else "en"

st.set_page_config(page_title="US K-line Guess Game", layout="wide")

# language selector on sidebar
st.sidebar.header("UI")
lang_label = st.sidebar.selectbox(TEXT[default_lang]["language"], ["English", "中文"], index=(0 if default_lang == "en" else 1))
lang = "en" if lang_label == "English" else "zh"
os.environ["UKG_LANG"] = lang

st.title(tr(lang, "title"))

# ---- Sidebar ----
st.sidebar.header(tr(lang, "game_settings"))

provider = st.sidebar.selectbox(tr(lang, "data_provider"), ["sample", "yfinance", "stooq"], index=0)
ticker = st.sidebar.text_input(tr(lang, "ticker"), value="AAPL").strip().upper()
lookback = st.sidebar.text_input(tr(lang, "lookback"), value="3y").strip()
bars = st.sidebar.slider(tr(lang, "bars_shown"), 40, 160, 80)
hide_n = st.sidebar.slider(tr(lang, "hidden_bars"), 1, 10, 5)
hint_level = st.sidebar.selectbox(tr(lang, "hint_level"), [1, 2, 3], index=1)
user_id = st.sidebar.text_input(tr(lang, "user_id"), value="renjun").strip()

st.sidebar.markdown("---")
st.sidebar.subheader(tr(lang, "indicators"))
ma_periods = st.sidebar.multiselect(tr(lang, "ma_periods"), [5, 10, 20, 50, 100], default=[20, 50])
rsi_period = st.sidebar.slider(tr(lang, "rsi_period"), 5, 30, 14)
show_volume = st.sidebar.checkbox(tr(lang, "show_volume"), value=True)
show_rsi = st.sidebar.checkbox(tr(lang, "show_rsi"), value=True)

st.sidebar.markdown("---")
st.sidebar.subheader(tr(lang, "llm"))
use_llm_default = os.getenv("UKG_USE_LLM", "0") == "1"
use_llm = st.sidebar.checkbox(tr(lang, "use_llm"), value=use_llm_default)

if use_llm:
    if not os.getenv("GOOGLE_API_KEY"):
        st.sidebar.warning(tr(lang, "missing_key"))
    if GeminiClient is None or LLMExplainAgent is None:
        st.sidebar.warning(tr(lang, "missing_llm_code"))

st.sidebar.caption(tr(lang, "disclaimer"))

# ---- Settings & agents ----
os.environ["UKG_DATA_PROVIDER"] = provider
os.environ["UKG_LOOKBACK"] = lookback

settings = get_settings()
episode_store = EpisodeStore(settings.episodes_dir)
score_store = ScoreStore(settings.data_dir / "scores.json")

data_agent = DataAgent.from_settings(settings)
episode_agent = EpisodeAgent(settings=settings, data_agent=data_agent)
hint_agent = HintAgent(settings=settings, data_agent=data_agent)
scoring_agent = ScoringAgent(episode_store=episode_store, score_store=score_store)

llm_agent = None
if use_llm and os.getenv("GOOGLE_API_KEY") and GeminiClient and LLMExplainAgent:
    try:
        llm_agent = LLMExplainAgent(hint_agent=hint_agent, client=GeminiClient())
    except Exception:
        llm_agent = None

# ---- Session state ----
if "private_ep" not in st.session_state:
    st.session_state.private_ep = None
if "last_hint" not in st.session_state:
    st.session_state.last_hint = None
if "last_hint_key" not in st.session_state:
    st.session_state.last_hint_key = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None
if "last_expl_key" not in st.session_state:
    st.session_state.last_expl_key = None


# ------------------------- Top buttons -------------------------
colA, colB, colC = st.columns([1, 1, 1])

with colA:
    if st.button(tr(lang, "new_episode"), width="stretch"):
        try:
            with st.spinner(tr(lang, "generating")):
                ep = episode_agent.generate_episode(
                    ticker=ticker,
                    timeframe="1d",
                    lookback=lookback,
                    bars=bars,
                    hide_n=hide_n,
                    seed=None,
                )
                episode_store.save(ep)
            st.session_state.private_ep = ep
            reset_round_state()
            st.success(f"{tr(lang, 'created')}: {ep.episode_id}")
        except Exception as e:
            st.error(str(e))

with colB:
    if st.button(tr(lang, "next"), width="stretch"):
        try:
            with st.spinner(tr(lang, "generating_next")):
                ep = episode_agent.generate_episode(
                    ticker=ticker,
                    timeframe="1d",
                    lookback=lookback,
                    bars=bars,
                    hide_n=hide_n,
                    seed=None,
                )
                episode_store.save(ep)
            st.session_state.private_ep = ep
            reset_round_state()
        except Exception as e:
            st.error(str(e))

with colC:
    if st.button(tr(lang, "reset"), width="stretch"):
        st.session_state.private_ep = None
        reset_round_state()


# ------------------------- Main -------------------------
ep = st.session_state.private_ep
if ep is None:
    st.info(tr(lang, "start_info"))
    st.stop()

left, right = st.columns([2, 1], gap="large")

# ------------------------- Chart panel -------------------------
with left:
    st.subheader(f"{tr(lang,'episode')}: {ep.episode_id}")
    st.caption(f"{ep.ticker} | timeframe={ep.timeframe} | shown={len(ep.candles)} | hidden={ep.hide_n} | provider={provider}")

    df_shown = candles_to_df(ep.candles)
    df_shown = add_indicators(df_shown, ma_periods=ma_periods, rsi_period=rsi_period)

    fig1 = plot_kline_with_indicators(
        df_shown,
        title=tr(lang, "shown_title"),
        ma_periods=ma_periods,
        rsi_period=rsi_period,
        show_volume=show_volume,
        show_rsi=show_rsi,
    )
    st.plotly_chart(fig1, width="stretch")

    if st.session_state.last_result is not None:
        truth = ep.hidden_truth
        df_all = candles_to_df(ep.candles + truth)
        df_all = add_indicators(df_all, ma_periods=ma_periods, rsi_period=rsi_period)

        fig2 = plot_kline_with_indicators(
            df_all,
            title=tr(lang, "revealed_title"),
            ma_periods=ma_periods,
            rsi_period=rsi_period,
            show_volume=show_volume,
            show_rsi=show_rsi,
        )
        st.plotly_chart(fig2, width="stretch")


# ------------------------- Interaction panel -------------------------
with right:
    st.subheader(tr(lang, "question"))

    # Question text (force consistent language)
    if lang == "en":
        question_text = f"Guess the overall direction of the next {ep.hide_n} bars (UP/DOWN)."
    else:
        question_text = f"猜未来 {ep.hide_n} 根K线整体方向（UP/DOWN）。"
    st.write(question_text)

    st.markdown("---")
    st.markdown(f"### {tr(lang, 'snapshot')}")

    last = df_shown.iloc[-1]
    last_close = float(last["c"])
    last_vol = float(last["v"])

    rsi_col = f"rsi{rsi_period}"
    last_rsi = float(last[rsi_col]) if rsi_col in df_shown.columns and pd.notna(last[rsi_col]) else float("nan")

    def _last_ma(p: int) -> float:
        col = f"ma{p}"
        return float(last[col]) if col in df_shown.columns and pd.notna(last[col]) else float("nan")

    ma20 = _last_ma(20)
    ma50 = _last_ma(50)

    vol_ratio = (
        float(df_shown["v"].tail(5).mean() / (df_shown["v"].tail(20).mean() + 1e-12))
        if len(df_shown) >= 20
        else float("nan")
    )

    c1, c2 = st.columns(2)
    c1.metric(tr(lang, "close"), f"{last_close:.2f}")
    c2.metric(tr(lang, "vol"), f"{last_vol:.0f}")

    c3, c4 = st.columns(2)
    c3.metric(f"RSI{rsi_period}", "—" if not pd.notna(last_rsi) else f"{last_rsi:.1f}")
    c4.metric(tr(lang, "vol_ratio"), "—" if not pd.notna(vol_ratio) else f"{vol_ratio:.2f}x")

    c5, c6 = st.columns(2)
    c5.metric("MA20", "—" if not pd.notna(ma20) else f"{ma20:.2f}")
    c6.metric("MA50", "—" if not pd.notna(ma50) else f"{ma50:.2f}")

    # Hint
    st.markdown("---")
    st.markdown(f"### {tr(lang, 'hint')}")
    if st.button(f"{tr(lang,'get_hint')} (L{hint_level})", width="stretch"):
        hint_key = (ep.episode_id, hint_level, bool(llm_agent), lang)
        if st.session_state.last_hint_key != hint_key:
            if llm_agent:
                with st.spinner("Gemini is generating..." if lang == "en" else "Gemini 正在生成提示..."):
                    st.session_state.last_hint = llm_agent.hint(ep, level=hint_level)
            else:
                h = hint_agent.get_hint(ep, level=hint_level)
                st.session_state.last_hint = h.hint
            st.session_state.last_hint_key = hint_key

    if st.session_state.last_hint:
        with st.expander(tr(lang, "view_hint"), expanded=True):
            st.write(st.session_state.last_hint)

    # Guess
    st.markdown("---")
    st.markdown(f"### {tr(lang, 'your_guess')}")
    disabled = st.session_state.last_result is not None
    up = st.button(tr(lang, "up"), width="stretch", disabled=disabled)
    down = st.button(tr(lang, "down"), width="stretch", disabled=disabled)

    if (up or down) and st.session_state.last_result is None:
        ans = "UP" if up else "DOWN"
        req = SubmitAnswerRequest(episode_id=ep.episode_id, user_id=user_id or "anonymous", answer=ans)  # type: ignore
        resp = scoring_agent.submit(req, reveal=True)
        st.session_state.last_result = resp.model_dump()
        st.session_state.last_explanation = None
        st.session_state.last_expl_key = None

    if st.session_state.last_result:
        res = st.session_state.last_result

        st.markdown("---")
        st.markdown(f"## {tr(lang, 'result')}")

        if res["correct"]:
            st.success(tr(lang, "correct"))
        else:
            st.error(tr(lang, "wrong"))

        r1, r2 = st.columns(2)
        r1.metric(tr(lang, "your_guess_label"), res["answer"])
        r2.metric(tr(lang, "truth"), res["truth_direction"])

        r3, r4 = st.columns(2)
        r3.metric(tr(lang, "score_delta"), res["score_delta"])
        r4.metric(tr(lang, "total_score"), res["user_score"])

        st.markdown(f"## {tr(lang, 'explanation')}")
        if llm_agent is None:
            st.info(tr(lang, "llm_unavailable"))
        else:
            expl_key = (ep.episode_id, res["answer"], res["truth_direction"], lang)
            if st.session_state.last_expl_key != expl_key or st.session_state.last_explanation is None:
                with st.spinner("Gemini is generating post-mortem..." if lang == "en" else "Gemini 正在生成复盘解释..."):
                    try:
                        st.session_state.last_explanation = llm_agent.post_mortem(ep, user_answer=res["answer"])
                    except Exception as e:
                        st.session_state.last_explanation = f"LLM failed: {e}" if lang == "en" else f"LLM 解释生成失败：{e}"
                    st.session_state.last_expl_key = expl_key

            with st.expander(tr(lang, "view_expl"), expanded=True):
                st.write(st.session_state.last_explanation)

        with st.expander(tr(lang, "debug")):
            st.json(res)

    st.markdown("---")
    st.markdown(f"### {tr(lang, 'leaderboard')}")
    top = score_store.top(10)
    lb_df = format_leaderboard(top, current_user=user_id if user_id else None, lang=lang)
    st.table(lb_df)

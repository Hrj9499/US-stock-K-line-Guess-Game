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
    out = df.copy()
    close = out["c"]

    for p in sorted(set(int(x) for x in ma_periods)):
        if p <= 1:
            continue
        out[f"ma{p}"] = close.rolling(p).mean()

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
    rows = 1 + int(show_volume) + int(show_rsi)

    if rows == 3:
        row_heights = [0.62, 0.20, 0.18]
    elif rows == 2:
        row_heights = [0.75, 0.25]
    else:
        row_heights = [1.0]

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
        fig.add_trace(
            go.Bar(x=df["t"], y=df["v"], name="Volume"),
            row=cur_row,
            col=1,
        )

    # RSI row
    if show_rsi:
        cur_row += 1
        rsi_col = f"rsi{rsi_period}"
        if rsi_col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["t"], y=df[rsi_col], mode="lines", name=f"RSI{rsi_period}"),
                row=cur_row,
                col=1,
            )
            try:
                fig.add_hline(y=70, line_dash="dash", row=cur_row, col=1)
                fig.add_hline(y=30, line_dash="dash", row=cur_row, col=1)
            except Exception:
                pass
            fig.update_yaxes(range=[0, 100], row=cur_row, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=680,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def format_leaderboard(top: list[tuple[str, int]], current_user: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(top, columns=["user_id", "score"])
    df.insert(0, "rank", range(1, len(df) + 1))
    if current_user:
        df["you"] = df["user_id"].apply(lambda x: "👤" if x == current_user else "")
        df = df[["rank", "you", "user_id", "score"]]
    return df


def reset_round_state() -> None:
    st.session_state.last_hint = None
    st.session_state.last_hint_key = None
    st.session_state.last_result = None
    st.session_state.last_explanation = None
    st.session_state.last_expl_key = None


# ------------------------- Page setup -------------------------
st.set_page_config(page_title="US K-line Guess Game", layout="wide")
st.title("📈 US Stock K-line Guess Game")

# ---- Sidebar ----
st.sidebar.header("Game Settings")

provider = st.sidebar.selectbox("Data provider", ["sample", "yfinance", "stooq"], index=0)
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
lookback = st.sidebar.text_input("Lookback", value="3y").strip()
bars = st.sidebar.slider("Bars shown", 40, 160, 80)
hide_n = st.sidebar.slider("Hidden bars", 1, 10, 5)
hint_level = st.sidebar.selectbox("Hint level", [1, 2, 3], index=1)
user_id = st.sidebar.text_input("User ID", value="renjun").strip()

st.sidebar.markdown("---")
st.sidebar.subheader("Indicators")
ma_periods = st.sidebar.multiselect("MA periods", [5, 10, 20, 50, 100], default=[20, 50])
rsi_period = st.sidebar.slider("RSI period", 5, 30, 14)
show_volume = st.sidebar.checkbox("Show volume", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("LLM (Gemini)")
use_llm_default = os.getenv("UKG_USE_LLM", "0") == "1"
use_llm = st.sidebar.checkbox("Use LLM explanation", value=use_llm_default)

if use_llm:
    if not os.getenv("GOOGLE_API_KEY"):
        st.sidebar.warning("Missing GOOGLE_API_KEY. LLM 已启用但无法调用。")
    if GeminiClient is None or LLMExplainAgent is None:
        st.sidebar.warning("LLM code not found/import failed. 请确认已添加 gemini_client.py 和 llm_explain_agent.py。")

st.sidebar.caption("⚠️ 本项目用于训练/娱乐，不构成投资建议。")

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
if "episode_id" not in st.session_state:
    st.session_state.episode_id = None
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
    if st.button("🎲 New Episode", width="stretch"):
        try:
            with st.spinner("Generating episode..."):
                ep = episode_agent.generate_episode(
                    ticker=ticker,
                    timeframe="1d",
                    lookback=lookback,
                    bars=bars,
                    hide_n=hide_n,
                    seed=None,
                )
                episode_store.save(ep)
            st.session_state.episode_id = ep.episode_id
            st.session_state.private_ep = ep
            reset_round_state()
            st.success(f"Created: {ep.episode_id}")
        except Exception as e:
            st.error(str(e))

with colB:
    if st.button("⏭️ Next", width="stretch"):
        try:
            with st.spinner("Generating next episode..."):
                ep = episode_agent.generate_episode(
                    ticker=ticker,
                    timeframe="1d",
                    lookback=lookback,
                    bars=bars,
                    hide_n=hide_n,
                    seed=None,
                )
                episode_store.save(ep)
            st.session_state.episode_id = ep.episode_id
            st.session_state.private_ep = ep
            reset_round_state()
        except Exception as e:
            st.error(str(e))

with colC:
    if st.button("🧹 Reset", width="stretch"):
        st.session_state.episode_id = None
        st.session_state.private_ep = None
        reset_round_state()


# ------------------------- Main -------------------------
ep = st.session_state.private_ep
if ep is None:
    st.info("左上点 **New Episode** 开始。建议先用 provider=sample（离线可跑）。")
    st.stop()

left, right = st.columns([2, 1], gap="large")

# ------------------------- Chart panel -------------------------
with left:
    st.subheader(f"Episode: {ep.episode_id}")
    st.caption(f"{ep.ticker} | timeframe={ep.timeframe} | shown={len(ep.candles)} | hidden={ep.hide_n} | provider={provider}")

    df_shown = candles_to_df(ep.candles)
    df_shown = add_indicators(df_shown, ma_periods=ma_periods, rsi_period=rsi_period)

    fig1 = plot_kline_with_indicators(
        df_shown,
        title="Shown candles + indicators",
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
            title="Revealed (shown + hidden) + indicators",
            ma_periods=ma_periods,
            rsi_period=rsi_period,
            show_volume=show_volume,
            show_rsi=show_rsi,
        )
        st.plotly_chart(fig2, width="stretch")


# ------------------------- Interaction panel -------------------------
with right:
    st.subheader("🎯 Question")
    st.write(ep.question)

    st.markdown("---")
    st.markdown("### 📌 Snapshot (last shown bar)")

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

    vol_ratio = float(df_shown["v"].tail(5).mean() / (df_shown["v"].tail(20).mean() + 1e-12)) if len(df_shown) >= 20 else float("nan")

    c1, c2 = st.columns(2)
    c1.metric("Close", f"{last_close:.2f}")
    c2.metric("Vol", f"{last_vol:.0f}")

    c3, c4 = st.columns(2)
    c3.metric(f"RSI{rsi_period}", "—" if not pd.notna(last_rsi) else f"{last_rsi:.1f}")
    c4.metric("Vol(5/20)", "—" if not pd.notna(vol_ratio) else f"{vol_ratio:.2f}x")

    c5, c6 = st.columns(2)
    c5.metric("MA20", "—" if not pd.notna(ma20) else f"{ma20:.2f}")
    c6.metric("MA50", "—" if not pd.notna(ma50) else f"{ma50:.2f}")

    # Hint
    st.markdown("---")
    st.markdown("### 💡 Hint")
    if st.button(f"Get Hint (L{hint_level})", width="stretch"):
        hint_key = (ep.episode_id, hint_level, bool(llm_agent))
        if st.session_state.last_hint_key != hint_key:
            if llm_agent:
                with st.spinner("Gemini 正在生成提示..."):
                    st.session_state.last_hint = llm_agent.hint(ep, level=hint_level)
            else:
                h = hint_agent.get_hint(ep, level=hint_level)
                st.session_state.last_hint = h.hint
            st.session_state.last_hint_key = hint_key

    if st.session_state.last_hint:
        with st.expander("查看提示", expanded=True):
            st.write(st.session_state.last_hint)

    # Guess
    st.markdown("---")
    st.markdown("### 🎮 Your Guess")
    disabled = st.session_state.last_result is not None
    up = st.button("🟢 UP", width="stretch", disabled=disabled)
    down = st.button("🔴 DOWN", width="stretch", disabled=disabled)

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
        st.markdown("## 🧾 Result")

        if res["correct"]:
            st.success("✅ Correct! 你猜对了")
        else:
            st.error("❌ Wrong 你猜错了")

        r1, r2 = st.columns(2)
        r1.metric("Your Guess", res["answer"])
        r2.metric("Truth", res["truth_direction"])

        r3, r4 = st.columns(2)
        r3.metric("Score Δ", res["score_delta"])
        r4.metric("Total Score", res["user_score"])

        st.markdown("## 🧠 Explanation (LLM)")
        if llm_agent is None:
            st.info("LLM 未启用或不可用（检查：UKG_USE_LLM=1、GOOGLE_API_KEY、以及 llm 文件/依赖）。")
        else:
            expl_key = (ep.episode_id, res["answer"], res["truth_direction"])
            if st.session_state.last_expl_key != expl_key or st.session_state.last_explanation is None:
                with st.spinner("Gemini 正在生成复盘解释..."):
                    try:
                        st.session_state.last_explanation = llm_agent.post_mortem(ep, user_answer=res["answer"])
                    except Exception as e:
                        st.session_state.last_explanation = f"LLM 解释生成失败：{e}"
                    st.session_state.last_expl_key = expl_key

            with st.expander("📌 查看复盘解释", expanded=True):
                st.write(st.session_state.last_explanation)

        with st.expander("Show raw details (debug)"):
            st.json(res)

    st.markdown("---")
    st.markdown("### 🏆 Leaderboard (Top 10)")
    top = score_store.top(10)
    lb_df = format_leaderboard(top, current_user=user_id if user_id else None)
    st.table(lb_df)

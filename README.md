# US K-Line Guess Game (Toy Agent Project)

A small “guess the next move” candlestick training game for US stocks.  
It generates an episode from historical OHLCV data, hides the last N bars, and asks you to guess **UP / DOWN**.  
The project can optionally use **Gemini (Google API)** to produce coach-style hints and post-mortem explanations.

## How this project is implemented as an “agent” system 

This repo is intentionally structured as a **small agentic system** rather than a single script.  
Each “agent” owns a clear responsibility and passes artifacts to the next stage, forming a loop:

**Observe → Retrieve/Compute → Reason/Explain → Act → Evaluate → Persist**

### Agent roles

- **DataAgent (Observe / Retrieve)**
  - Fetches OHLCV data using a pluggable **data provider** (`sample`, `yfinance`, `stooq`)
  - Normalizes data into a consistent schema used by downstream agents
  - Can cache results locally for fast iteration

- **EpisodeAgent (Episode generation)**
  - Samples a historical window (e.g. 80 bars)
  - Hides the last N bars as the “future”
  - Outputs an `Episode` artifact (stored as JSON) that includes:
    - shown candles, hidden truth candles, question metadata, and a deterministic seed

- **HintAgent (Reasoning primitives)**
  - Computes **rule-based** signals / summary statistics (trend, volatility, key levels)
  - Produces level-based hints (L1/L2/L3) in a structured way
  - Acts as the “toolbox” for both UI and LLM explanations

- **LLMExplainAgent (LLM reasoning & narration, optional)**
  - Uses Gemini to transform structured stats into a **coach-style** hint (without leaking the answer)
  - Generates a reveal-time **post-mortem** explanation after the user submits a guess
  - The system keeps the *decision logic* in deterministic agents and uses LLM mainly for **natural-language explanation**

- **ScoringAgent (Evaluate)**
  - Validates the user’s guess against hidden truth
  - Updates score and leaderboard
  - Persists scores to a local file store for a lightweight “game loop”

### Why this design

- **Modularity**: swap providers / swap hint logic / swap UI without rewriting everything  
- **Reproducibility**: episodes are artifacts (JSON) and can be replayed  
- **Extensibility**: easy to add new tasks (e.g., “breakout next 5 bars?”, “touch MA20?”)  
- **LLM safety**: LLM is used for explanation, not trading commands

## Features 

- **Episode generation**: sample a historical window and hide last N bars
- **Interactive UI** (Streamlit): candlestick chart + buttons (UP/DOWN)
- **Indicators**: MA lines, Volume, RSI
- **Scoring & leaderboard**
- **LLM explanation (optional)**: Gemini generates hints + post-mortem explanations
- **Bilingual UI**: English / Chinese toggle

## Quick Start 

### 1) Install
```bash
pip install -e .
pip install -U streamlit plotly google-genai yfinance
```

### 2) Run (offline sample data)
```bash
export UKG_DATA_PROVIDER=sample
streamlit run src/us_kline_guess/ui/streamlit_app.py
```

### 3) Run with Yahoo Finance
```bash
export UKG_DATA_PROVIDER=yfinance
streamlit run src/us_kline_guess/ui/streamlit_app.py
```

### 4) Enable Gemini (LLM hints + explanations)
```bash
export GOOGLE_API_KEY="YOUR_KEY"
export UKG_USE_LLM=1
export UKG_GEMINI_MODEL="gemini-2.5-flash"
streamlit run src/us_kline_guess/ui/streamlit_app.py
```

## Environment Variables

- `UKG_DATA_PROVIDER`: `sample` | `yfinance` | `stooq`
- `GOOGLE_API_KEY`: Gemini API key
- `UKG_USE_LLM`: `1` to enable LLM explanation
- `UKG_GEMINI_MODEL`: e.g. `gemini-2.5-flash`
- `UKG_LANG`: `en` or `zh` (UI also sets this automatically)


## Attribution / Usage Note 

You are welcome to use this toy project for learning, demos, and experiments.
Please keep a short attribution note that this project was created by the owner of this GitHub account (HRJ9499).

---


# 美股 K 线猜猜乐（Agent 小项目）

这是一个小型“猜涨跌”的 K 线训练小游戏：  
系统会从历史行情中截取一段 OHLCV 数据，隐藏最后 N 根 K 线，让你猜 **UP / DOWN**。  
项目支持接入 **Gemini（Google API）**，生成教练式提示与揭晓后的复盘解释。

## 功能特点（中文）

- **关卡生成**：从历史数据抽样一段走势，隐藏最后 N 根
- **交互 UI（Streamlit）**：K 线图 + UP/DOWN 按钮
- **指标**：均线 MA、成交量 Volume、RSI
- **计分与排行榜**
- **LLM 解释（可选）**：Gemini 生成提示 + 复盘解释
- **中英文 UI 切换**

## 快速开始（中文）

### 1）安装
```bash
pip install -e .
pip install -U streamlit plotly google-genai yfinance
```

### 2）离线运行（sample 数据）
```bash
export UKG_DATA_PROVIDER=sample
streamlit run src/us_kline_guess/ui/streamlit_app.py
```

### 3）使用 Yahoo Finance 数据
```bash
export UKG_DATA_PROVIDER=yfinance
streamlit run src/us_kline_guess/ui/streamlit_app.py
```

### 4）启用 Gemini（提示 + 复盘解释）
```bash
export GOOGLE_API_KEY="你的key"
export UKG_USE_LLM=1
export UKG_GEMINI_MODEL="gemini-2.5-flash"
streamlit run src/us_kline_guess/ui/streamlit_app.py
```

## 环境变量（中文）

- `UKG_DATA_PROVIDER`：`sample` | `yfinance` | `stooq`
- `GOOGLE_API_KEY`：Gemini API key
- `UKG_USE_LLM`：设为 `1` 启用 LLM 解释
- `UKG_GEMINI_MODEL`：例如 `gemini-2.5-flash`
- `UKG_LANG`：`en` 或 `zh`（UI 会自动设置）

> ⚠️ 仅用于训练/娱乐，不构成投资建议。

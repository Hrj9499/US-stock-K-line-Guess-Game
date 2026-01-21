# us-kline-guess-agent (Toy)

一个**可直接跑起来**的小型“agent 风格”项目：用免费数据源拉取美股历史 K 线，生成“猜未来 N 根方向”的关卡（episode），并提供一个 **RAG-ish 的相似形态检索提示**（不依赖任何付费 LLM / API Key）。

> 免责声明：本项目仅用于学习/游戏/研究，不构成任何投资建议。

---

## 你能得到什么？

- ✅ **Data Agent**：从免费数据源获取 OHLCV（默认 `yfinance`，可切换 `stooq`，离线可用 `sample`）
- ✅ **Episode Agent**：随机切片生成关卡（隐藏最后 N 根）
- ✅ **Hint Agent**：
  - L1：趋势/波动提示
  - L2：区间/关键位提示
  - L3：**相似形态检索** → 统计未来 N 根上涨概率、均值、分位数（类似“检索增强”）
- ✅ **Scoring Agent**：提交答案、计分、排行榜
- ✅ **FastAPI 服务** + **终端小游戏 CLI**

---

## 项目结构（建议 GitHub repo 直接这么放）

```
us-kline-guess-agent/
  src/us_kline_guess/
    api.py                 # FastAPI 入口
    cli.py                 # 终端入口（generate/play）
    config.py
    models.py
    agents/
      data_agent.py
      episode_agent.py
      hint_agent.py
      scoring_agent.py
    providers/
      base.py
      yfinance_provider.py
      stooq_provider.py
      sample_provider.py
    storage/
      file_store.py
  data/
    cache/                 # 缓存（CSV）
    episodes/              # 生成的关卡（JSON）
    scores.json            # 计分（JSON）
  sample_data/             # 离线样例数据（CSV）
```

---

## 数据来源选择（免费）

### 1) 默认：yfinance（Yahoo Finance 抓取）
- 优点：用起来最简单、社区用得多
- 缺点：可能被限流；属于抓取方式，稳定性一般

### 2) 备选：Stooq（免费 CSV 接口）
- 无需 Key：`https://stooq.com/q/d/l/?s=aapl.us&i=d`
- 优点：完全免费，接口简单
- 缺点：符号映射有时麻烦（比如 `BRK-B`）

### 3) 离线：sample（项目自带 CSV）
- 方便你在无网络环境也能跑通流程

切换数据源：
```bash
export UKG_DATA_PROVIDER=stooq   # 或 yfinance / sample
```

---

## 快速开始（推荐）

### 1) 安装
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -U pip
pip install -e .
```

### 2) 启动 API
```bash
uvicorn us_kline_guess.api:app --reload --port 8000
```

打开：
- Swagger: http://127.0.0.1:8000/docs

### 3) 生成一关（API）
```bash
curl -X POST "http://127.0.0.1:8000/episodes/generate" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","timeframe":"1d","lookback":"3y","bars":80,"hide_n":5}'
```

### 4) 获取提示（API）
```bash
curl "http://127.0.0.1:8000/episodes/<episode_id>/hint?level=3"
```

### 5) 提交答案（API）
```bash
curl -X POST "http://127.0.0.1:8000/answers/submit" \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"<episode_id>","user_id":"alice","answer":"UP"}'
```

---

## 终端直接玩（无前端也能跑）

离线跑（强制 sample 数据）：
```bash
export UKG_DATA_PROVIDER=sample
us-kline-guess play --ticker AAPL --hint-level 3
```

在线跑（默认 yfinance）：
```bash
export UKG_DATA_PROVIDER=yfinance
us-kline-guess play --ticker TSLA --hint-level 2
```

---

## 环境变量

| 变量 | 默认 | 说明 |
|---|---:|---|
| `UKG_DATA_PROVIDER` | `yfinance` | `yfinance` / `stooq` / `sample` |
| `UKG_LOOKBACK` | `3y` | 数据回看范围（近似） |
| `UKG_BARS` | `80` | 展示 K 线根数 |
| `UKG_HIDE_N` | `5` | 隐藏根数 |
| `UKG_SIMILAR_TOPK` | `20` | 相似检索 top-k |
| `UKG_OFFLINE_OK` | `true` | 数据源失败是否允许降级到 sample |

---

## 下一步你可以怎么扩展

- 做成微信小程序 / Web 前端（ECharts/TradingView 绘制 candles）
- 加更多题型：猜波动区间、猜破位、猜 ticker
- 把 Hint Agent 升级成真正的 LLM Agent（把相似片段作为 RAG Context）
- 加评测：不同 bars/hide_n/topk 对玩家胜率影响

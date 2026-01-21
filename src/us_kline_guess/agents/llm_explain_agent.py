from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict

from us_kline_guess.models import EpisodePrivate
from us_kline_guess.agents.hint_agent import HintAgent
from us_kline_guess.llm.gemini_client import GeminiClient


SYSTEM_EN = """You are a coach for a candlestick-reading training game, NOT a financial advisor.
Rules:
- Do NOT give trading instructions (no "buy/sell/long/short/open/close positions")
- Only explain patterns, probabilities, uncertainty, and risk
- Output must be in English
- Hint stage must NOT reveal the answer
"""

SYSTEM_ZH = """你是一个“股票K线训练小游戏”的教练，不是投资顾问。
要求：
- 不要给出买卖建议，不要出现“买入/卖出/做多/做空/开仓/平仓”等指令
- 只做形态解读、概率与不确定性解释
- 输出中文，条理清楚，短句为主
- 提示阶段不要泄露答案
"""


def _lang() -> str:
    lang = os.getenv("UKG_LANG", "en").lower()
    return "zh" if lang.startswith("zh") else "en"


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _clip_text(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...(truncated, {len(s)} chars total)"


def _clip_stats(stats: Any, max_chars: int = 8000) -> Any:
    try:
        txt = _safe_json(stats)
    except Exception:
        txt = str(stats)
    return _clip_text(txt, max_chars)


@dataclass
class LLMExplainAgent:
    hint_agent: HintAgent
    client: GeminiClient

    def _system(self) -> str:
        return SYSTEM_ZH if _lang() == "zh" else SYSTEM_EN

    def hint(self, ep: EpisodePrivate, level: int) -> str:
        h = self.hint_agent.get_hint(ep, level=level)

        payload: Dict[str, Any] = {
            "type": "hint",
            "level": level,
            "question": ep.question,
            "ticker": ep.ticker,
            "timeframe": ep.timeframe,
            "hint_text_rule_based": h.hint,
            "stats_clipped": _clip_stats(h.stats, max_chars=6000),
        }

        if _lang() == "zh":
            prompt = (
                self._system()
                + "\n\n任务：把下面信息改写成更自然的“教练式提示”。\n"
                  "要求：\n"
                  "1) 不要直接给出答案（不要说一定UP/DOWN）。\n"
                  "2) level=1轻提示，level=2中提示，level=3强提示（仍不直接给答案）。\n"
                  "3) 给 3-6 条 bullet points，最后用一句话提醒不确定性。\n\n"
                + "输入(JSON)：\n"
                + _safe_json(payload)
            )
        else:
            prompt = (
                self._system()
                + "\n\nTask: Rewrite the following into a natural, coach-style hint.\n"
                  "Requirements:\n"
                  "1) Do NOT reveal the answer (do not say it must be UP/DOWN).\n"
                  "2) level=1 light hint, level=2 medium hint, level=3 stronger hint (still no direct answer).\n"
                  "3) Output 3-6 bullet points, ending with a 1-sentence uncertainty reminder.\n\n"
                + "Input(JSON):\n"
                + _safe_json(payload)
            )

        return self.client.generate(prompt)

    def post_mortem(self, ep: EpisodePrivate, user_answer: str) -> str:
        h1 = self.hint_agent.get_hint(ep, level=1)
        h2 = self.hint_agent.get_hint(ep, level=2)
        h3 = self.hint_agent.get_hint(ep, level=3)

        payload: Dict[str, Any] = {
            "type": "post_mortem",
            "question": ep.question,
            "ticker": ep.ticker,
            "timeframe": ep.timeframe,
            "user_answer": user_answer,
            "truth_direction": ep.truth_direction,
            "hint_L1": {"text": h1.hint, "stats_clipped": _clip_stats(h1.stats, max_chars=3000)},
            "hint_L2": {"text": h2.hint, "stats_clipped": _clip_stats(h2.stats, max_chars=3000)},
            "hint_L3": {"text": h3.hint, "stats_clipped": _clip_stats(h3.stats, max_chars=6000)},
        }

        if _lang() == "zh":
            prompt = (
                self._system()
                + "\n\n任务：做一段“揭晓后的复盘解释”。\n"
                  "要求：\n"
                  "1) 先用一句话给结论：最终为何是 UP/DOWN。\n"
                  "2) 然后分 3-5 条解释：趋势/关键位/波动/量能/相似片段统计（若有）。\n"
                  "3) 再写一条：为什么用户容易猜错（常见误区）。\n"
                  "4) 禁止交易建议，只讲解读与不确定性。\n\n"
                + "输入(JSON)：\n"
                + _safe_json(payload)
            )
        else:
            prompt = (
                self._system()
                + "\n\nTask: Write a reveal-time post-mortem explanation.\n"
                  "Requirements:\n"
                  "1) Start with one sentence conclusion: why the final direction is UP/DOWN.\n"
                  "2) Then 3-5 bullet points: trend / key levels / volatility / volume / similar-pattern stats (if any).\n"
                  "3) Add one bullet: common mistakes (why users guess wrong).\n"
                  "4) No trading advice, only interpretation & uncertainty.\n\n"
                + "Input(JSON):\n"
                + _safe_json(payload)
            )

        return self.client.generate(prompt)

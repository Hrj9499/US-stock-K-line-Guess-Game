from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Candle(BaseModel):
    t: str = Field(..., description="Timestamp (YYYY-MM-DD for 1d timeframe)")
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0


AnswerDirection = Literal["UP", "DOWN"]


class EpisodePublic(BaseModel):
    episode_id: str
    ticker: str
    timeframe: str
    bars: int
    hide_n: int
    created_at: str

    start: str
    end: str

    candles: list[Candle] = Field(..., description="Shown candles (truth hidden)")
    question: str
    answer_type: str
    meta: dict[str, Any] = Field(default_factory=dict)


class EpisodePrivate(EpisodePublic):
    hidden_truth: list[Candle]
    truth_direction: AnswerDirection


class GenerateEpisodeRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL", "TSLA"])
    timeframe: str = Field(default="1d", description="Currently only 1d is supported in the toy system")
    lookback: str = Field(default="3y", description="Provider lookback string, e.g., 1y/3y/5y/max")
    bars: int = Field(default=80, ge=20, le=250)
    hide_n: int = Field(default=5, ge=1, le=30)
    seed: Optional[int] = Field(default=None, description="For reproducibility")


class HintResponse(BaseModel):
    episode_id: str
    level: int
    title: str
    hint: str
    stats: dict[str, Any] = Field(default_factory=dict)


class SubmitAnswerRequest(BaseModel):
    episode_id: str
    user_id: str = Field(default="anonymous")
    answer: AnswerDirection


class SubmitAnswerResponse(BaseModel):
    episode_id: str
    user_id: str
    answer: AnswerDirection
    correct: bool
    truth_direction: AnswerDirection
    score_delta: int
    user_score: int
    revealed: dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntry(BaseModel):
    user_id: str
    score: int


class LeaderboardResponse(BaseModel):
    updated_at: str
    top: list[LeaderboardEntry]

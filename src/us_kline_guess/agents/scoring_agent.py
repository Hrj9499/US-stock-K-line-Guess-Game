from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import SubmitAnswerRequest, SubmitAnswerResponse
from ..storage.file_store import EpisodeStore, ScoreStore


@dataclass
class ScoringAgent:
    episode_store: EpisodeStore
    score_store: ScoreStore

    def submit(self, req: SubmitAnswerRequest, reveal: bool = True) -> SubmitAnswerResponse:
        ep = self.episode_store.load_private(req.episode_id)

        correct = (req.answer == ep.truth_direction)
        score_delta = 10 if correct else -2
        user_score = self.score_store.add_score(req.user_id, score_delta)

        revealed: dict[str, Any] = {}
        if reveal:
            revealed = {
                "hidden_truth": [c.model_dump() for c in ep.hidden_truth],
                "truth_direction": ep.truth_direction,
                "start": ep.start,
                "end": ep.end,
            }

        return SubmitAnswerResponse(
            episode_id=req.episode_id,
            user_id=req.user_id,
            answer=req.answer,
            correct=correct,
            truth_direction=ep.truth_direction,
            score_delta=score_delta,
            user_score=user_score,
            revealed=revealed,
        )

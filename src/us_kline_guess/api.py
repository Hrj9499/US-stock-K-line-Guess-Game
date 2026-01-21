from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .config import get_settings
from .models import (
    EpisodePublic,
    GenerateEpisodeRequest,
    HintResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    LeaderboardResponse,
    LeaderboardEntry,
)
from .agents.data_agent import DataAgent
from .agents.episode_agent import EpisodeAgent
from .agents.hint_agent import HintAgent
from .agents.scoring_agent import ScoringAgent
from .storage.file_store import EpisodeStore, ScoreStore


load_dotenv()
settings = get_settings()

episode_store = EpisodeStore(settings.episodes_dir)
score_store = ScoreStore(settings.data_dir / "scores.json")

data_agent = DataAgent.from_settings(settings)
episode_agent = EpisodeAgent(settings=settings, data_agent=data_agent)
hint_agent = HintAgent(settings=settings, data_agent=data_agent)
scoring_agent = ScoringAgent(episode_store=episode_store, score_store=score_store)

app = FastAPI(
    title="US K-line Guess Agent (Toy)",
    version="0.1.0",
    description="A minimal, runnable 'agent-like' RAG-ish K-line guessing backend (free data providers).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # toy only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "us-kline-guess-agent",
        "status": "ok",
        "docs": "/docs",
        "provider": settings.data_provider,
    }


@app.get("/health")
def health():
    return {"ok": True, "provider": settings.data_provider}


@app.post("/episodes/generate", response_model=EpisodePublic)
def generate_episode(req: GenerateEpisodeRequest):
    try:
        ep = episode_agent.generate_episode(
            ticker=req.ticker,
            timeframe=req.timeframe,
            lookback=req.lookback,
            bars=req.bars,
            hide_n=req.hide_n,
            seed=req.seed,
        )
        episode_store.save(ep)
        return episode_store.load_public(ep.episode_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/episodes/{episode_id}", response_model=EpisodePublic)
def get_episode(episode_id: str):
    try:
        return episode_store.load_public(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="episode not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/episodes/{episode_id}/hint", response_model=HintResponse)
def get_hint(episode_id: str, level: int = Query(1, ge=1, le=3)):
    try:
        ep = episode_store.load_private(episode_id)
        return hint_agent.get_hint(ep, level=level)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="episode not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/episodes/{episode_id}/reveal")
def reveal_episode(episode_id: str):
    try:
        ep = episode_store.load_private(episode_id)
        return {
            "episode_id": ep.episode_id,
            "truth_direction": ep.truth_direction,
            "hidden_truth": [c.model_dump() for c in ep.hidden_truth],
            "meta": ep.meta,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="episode not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/episodes/recent")
def recent_episodes(limit: int = Query(20, ge=1, le=200)):
    return {"episodes": [e.model_dump() for e in episode_store.list_recent(limit=limit)]}


@app.post("/answers/submit", response_model=SubmitAnswerResponse)
def submit_answer(req: SubmitAnswerRequest):
    try:
        return scoring_agent.submit(req, reveal=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="episode not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/leaderboard", response_model=LeaderboardResponse)
def leaderboard(top: int = Query(10, ge=1, le=100)):
    items = score_store.top(n=top)
    return LeaderboardResponse(
        updated_at=score_store.updated_at(),
        top=[LeaderboardEntry(user_id=u, score=s) for u, s in items],
    )

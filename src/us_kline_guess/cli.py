from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import get_settings
from .agents.data_agent import DataAgent
from .agents.episode_agent import EpisodeAgent
from .agents.hint_agent import HintAgent
from .storage.file_store import EpisodeStore, ScoreStore
from .agents.scoring_agent import ScoringAgent
from .models import SubmitAnswerRequest


def cmd_generate(args: argparse.Namespace) -> int:
    settings = get_settings()
    data_agent = DataAgent.from_settings(settings)
    ep_agent = EpisodeAgent(settings=settings, data_agent=data_agent)

    ep = ep_agent.generate_episode(
        ticker=args.ticker,
        timeframe=args.timeframe,
        lookback=args.lookback,
        bars=args.bars,
        hide_n=args.hide_n,
        seed=args.seed,
    )
    store = EpisodeStore(settings.episodes_dir)
    store.save(ep)
    print(json.dumps(store.load_public(ep.episode_id).model_dump(), ensure_ascii=False, indent=2))
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    # A tiny terminal game loop (works even with sample provider)
    settings = get_settings()
    data_agent = DataAgent.from_settings(settings)
    ep_agent = EpisodeAgent(settings=settings, data_agent=data_agent)
    hint_agent = HintAgent(settings=settings, data_agent=data_agent)

    ep = ep_agent.generate_episode(
        ticker=args.ticker,
        timeframe=args.timeframe,
        lookback=args.lookback,
        bars=args.bars,
        hide_n=args.hide_n,
        seed=args.seed,
    )

    store = EpisodeStore(settings.episodes_dir)
    store.save(ep)

    print(f"Episode: {ep.episode_id}")
    print(f"Ticker: {ep.ticker}  Timeframe: {ep.timeframe}  Bars shown: {len(ep.candles)}  Hidden: {ep.hide_n}")
    print(f"Question: {ep.question}")
    print("")
    print("Last 5 shown candles (t, o, h, l, c):")
    for c in ep.candles[-5:]:
        print(f"  {c.t}  {c.o:.2f} {c.h:.2f} {c.l:.2f} {c.c:.2f}")

    if args.hint_level:
        h = hint_agent.get_hint(ep, level=args.hint_level)
        print("")
        print(f"[Hint L{h.level}] {h.title}")
        print(h.hint)

    ans = input("\nYour answer (UP/DOWN): ").strip().upper()
    if ans not in {"UP", "DOWN"}:
        print("Invalid answer. Use UP or DOWN.")
        return 2

    score_store = ScoreStore(settings.data_dir / "scores.json")
    scoring = ScoringAgent(episode_store=store, score_store=score_store)
    resp = scoring.submit(SubmitAnswerRequest(episode_id=ep.episode_id, user_id=args.user_id, answer=ans))  # type: ignore

    print("\nResult:")
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="us-kline-guess", description="US K-line Guess Agent (Toy)")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ticker", default="AAPL")
    common.add_argument("--timeframe", default=os.getenv("UKG_TIMEFRAME", "1d"))
    common.add_argument("--lookback", default=os.getenv("UKG_LOOKBACK", "3y"))
    common.add_argument("--bars", type=int, default=int(os.getenv("UKG_BARS", "80")))
    common.add_argument("--hide-n", type=int, default=int(os.getenv("UKG_HIDE_N", "5")))
    common.add_argument("--seed", type=int, default=None)

    g = sub.add_parser("generate", parents=[common], help="Generate an episode and print public JSON")
    g.set_defaults(func=cmd_generate)

    play = sub.add_parser("play", parents=[common], help="Play a round in terminal")
    play.add_argument("--user-id", default="anonymous")
    play.add_argument("--hint-level", type=int, default=0, help="0 to disable, or 1..3")
    play.set_defaults(func=cmd_play)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

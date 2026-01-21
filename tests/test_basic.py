import os

def test_generate_episode_with_sample_provider():
    os.environ["UKG_DATA_PROVIDER"] = "sample"
    from us_kline_guess.config import get_settings
    from us_kline_guess.agents.data_agent import DataAgent
    from us_kline_guess.agents.episode_agent import EpisodeAgent

    settings = get_settings()
    data_agent = DataAgent.from_settings(settings)
    ep_agent = EpisodeAgent(settings=settings, data_agent=data_agent)

    ep = ep_agent.generate_episode(ticker="AAPL", timeframe="1d", lookback="3y", bars=80, hide_n=5, seed=123)
    assert ep.episode_id
    assert len(ep.candles) == 80
    assert len(ep.hidden_truth) == 5
    assert ep.truth_direction in ("UP", "DOWN")

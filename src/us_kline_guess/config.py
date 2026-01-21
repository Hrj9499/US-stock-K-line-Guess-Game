from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # --- project paths ---
    project_root: Path
    data_dir: Path
    episodes_dir: Path
    cache_dir: Path
    sample_data_dir: Path

    # --- providers ---
    # "yfinance" (default) | "stooq" | "sample"
    data_provider: str

    # --- episode defaults ---
    default_timeframe: str
    default_lookback: str
    default_bars: int
    default_hide_n: int

    # --- hint/similarity ---
    similar_topk: int

    # --- runtime options ---
    offline_ok: bool


def get_settings() -> Settings:
    # Resolve repository root at runtime:
    #   src/us_kline_guess/config.py  -> repo root is 2 parents up
    project_root = Path(__file__).resolve().parents[2]

    data_dir = Path(os.getenv("UKG_DATA_DIR", str(project_root / "data"))).resolve()
    episodes_dir = data_dir / "episodes"
    cache_dir = data_dir / "cache"
    sample_data_dir = Path(os.getenv("UKG_SAMPLE_DATA_DIR", str(project_root / "sample_data"))).resolve()

    # Ensure dirs exist (safe for local dev)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        episodes_dir=episodes_dir,
        cache_dir=cache_dir,
        sample_data_dir=sample_data_dir,
        data_provider=os.getenv("UKG_DATA_PROVIDER", "yfinance"),
        default_timeframe=os.getenv("UKG_TIMEFRAME", "1d"),
        default_lookback=os.getenv("UKG_LOOKBACK", "3y"),
        default_bars=int(os.getenv("UKG_BARS", "80")),
        default_hide_n=int(os.getenv("UKG_HIDE_N", "5")),
        similar_topk=int(os.getenv("UKG_SIMILAR_TOPK", "20")),
        offline_ok=_env_bool("UKG_OFFLINE_OK", True),
    )

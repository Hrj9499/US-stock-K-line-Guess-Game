from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ..models import EpisodePrivate, EpisodePublic


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeStore:
    def __init__(self, episodes_dir: Path):
        self.episodes_dir = episodes_dir
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self._adapter_private = TypeAdapter(EpisodePrivate)
        self._adapter_public = TypeAdapter(EpisodePublic)

    def _path(self, episode_id: str) -> Path:
        safe = episode_id.replace("/", "_")
        return self.episodes_dir / f"{safe}.json"

    def save(self, ep: EpisodePrivate) -> None:
        path = self._path(ep.episode_id)
        payload = ep.model_dump()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_private(self, episode_id: str) -> EpisodePrivate:
        path = self._path(episode_id)
        if not path.exists():
            raise FileNotFoundError(f"episode not found: {episode_id}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        return self._adapter_private.validate_python(obj)

    def load_public(self, episode_id: str) -> EpisodePublic:
        ep = self.load_private(episode_id)
        obj = ep.model_dump()
        obj.pop("hidden_truth", None)
        obj.pop("truth_direction", None)
        return self._adapter_public.validate_python(obj)

    def list_recent(self, limit: int = 20) -> list[EpisodePublic]:
        paths = sorted(self.episodes_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out: list[EpisodePublic] = []
        for p in paths[:limit]:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                # convert to public
                obj.pop("hidden_truth", None)
                obj.pop("truth_direction", None)
                out.append(self._adapter_public.validate_python(obj))
            except Exception:
                continue
        return out


class ScoreStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"updated_at": _now_iso(), "scores": {}}, indent=2), encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        data["updated_at"] = _now_iso()
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_score(self, user_id: str) -> int:
        data = self._read()
        return int(data.get("scores", {}).get(user_id, 0))

    def add_score(self, user_id: str, delta: int) -> int:
        data = self._read()
        scores = data.setdefault("scores", {})
        scores[user_id] = int(scores.get(user_id, 0)) + int(delta)
        self._write(data)
        return int(scores[user_id])

    def top(self, n: int = 10) -> list[tuple[str, int]]:
        data = self._read()
        scores = data.get("scores", {})
        items = [(uid, int(score)) for uid, score in scores.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    def updated_at(self) -> str:
        return self._read().get("updated_at", _now_iso())

import json
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


@dataclass
class TraceStage:
    name: str
    started_at_ms: float
    ended_at_ms: float = 0.0
    duration_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def finish(self, ended_at_ms: float) -> None:
        self.ended_at_ms = ended_at_ms
        self.duration_ms = max(0.0, ended_at_ms - self.started_at_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "started_at_ms": round(self.started_at_ms, 3),
            "ended_at_ms": round(self.ended_at_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "meta": self.meta,
        }


class TraceContext:
    def __init__(self, user_id: str, conv_id: str, message: str):
        self.trace_id = uuid.uuid4().hex
        self.user_id = user_id
        self.conv_id = conv_id
        self.message = message
        self.started_at_ms = time.time() * 1000
        self.completed = False
        self.success = True
        self.error = ""
        self._stages: List[TraceStage] = []

    @contextmanager
    def stage(self, name: str, **meta: Any):
        stage = TraceStage(name=name, started_at_ms=time.perf_counter() * 1000, meta=dict(meta))
        self._stages.append(stage)
        try:
            yield stage
        except Exception as ex:
            stage.meta.setdefault("error", str(ex))
            raise
        finally:
            stage.finish(time.perf_counter() * 1000)

    def finalize(self, *, success: bool = True, error: str = "") -> None:
        self.completed = True
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        total_ms = 0.0
        if self._stages:
            total_ms = sum(stage.duration_ms for stage in self._stages if stage.name == "chat.total")
            if total_ms == 0.0:
                total_ms = sum(stage.duration_ms for stage in self._stages)
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "conv_id": self.conv_id,
            "message": self.message,
            "started_at_ms": round(self.started_at_ms, 3),
            "completed": self.completed,
            "success": self.success,
            "error": self.error,
            "total_ms": round(total_ms, 3),
            "stages": [stage.to_dict() for stage in self._stages],
        }


class TraceStore:
    def __init__(self, capacity: int = 200, jsonl_dir: Optional[str] = None):
        self._items: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._jsonl_dir = Path(jsonl_dir) if jsonl_dir else None
        if self._jsonl_dir is not None:
            self._jsonl_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: TraceContext) -> None:
        payload = trace.to_dict()
        self._items.appendleft(payload)
        self._index[payload["trace_id"]] = payload
        if self._jsonl_dir is not None:
            filename = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
            with (self._jsonl_dir / filename).open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        return self._index.get(trace_id)

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._items)[:limit]

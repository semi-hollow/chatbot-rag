import json
import time
import uuid
from pathlib import Path

from app.config import TRACE_FILE


class Trace:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.trace_id = uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self.stages: dict[str, dict] = {}
        self.payload: dict = {}

    def mark(self, stage: str, data: dict) -> None:
        if not self.enabled:
            return
        self.stages[stage] = data

    def add(self, key: str, value):
        if not self.enabled:
            return
        self.payload[key] = value

    def save(self) -> None:
        if not self.enabled:
            return
        record = {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "duration_ms": round((time.time() - self.started_at) * 1000, 2),
            "payload": self.payload,
            "stages": self.stages,
        }
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_traces(limit: int = 20) -> list[dict]:
    if not Path(TRACE_FILE).exists():
        return []
    with Path(TRACE_FILE).open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]

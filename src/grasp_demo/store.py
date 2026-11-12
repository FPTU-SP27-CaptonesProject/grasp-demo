"""Thread-safe event storage, SSE delivery, persistence and replay."""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator


@dataclass
class DemoRun:
    run_id: str
    request: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False
    changed: threading.Condition = field(default_factory=threading.Condition)

    def publish(self, event: dict[str, Any]) -> None:
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def close(self) -> None:
        with self.changed:
            self.closed = True
            self.changed.notify_all()

    def wait_from(self, index: int, timeout: float = 15.0) -> tuple[list[dict[str, Any]], bool]:
        """Events from `index` on (waiting up to `timeout` for one), and whether the run has closed."""
        with self.changed:
            if index >= len(self.events) and not self.closed:
                self.changed.wait(timeout)
            return list(self.events[index:]), self.closed


class RunStore:
    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, DemoRun] = {}
        self._lock = threading.Lock()

    def create(self, request: dict[str, Any]) -> DemoRun:
        run = DemoRun(uuid.uuid4().hex[:12], request)
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> DemoRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def persist(self, run: DemoRun) -> Path:
        created = time.strftime("%Y%m%d-%H%M%S", time.localtime(run.created_at))
        path = self.runs_dir / f"{created}-{run.run_id}.jsonl"
        with run.changed:
            events = list(run.events)
        rows = [{"type": "request", "data": run.request, "run_id": run.run_id, "created_at": run.created_at}, *events]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def history(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.runs_dir.glob("*.jsonl"), reverse=True):
            first = path.open(encoding="utf-8").readline()
            if not first:
                continue
            record = json.loads(first)
            result.append({"id": record.get("run_id"), "file": path.name, "request": record.get("data", {}), "created_at": record.get("created_at")})
        return result

    def replay(self, filename: str) -> list[dict[str, Any]]:
        target = (self.runs_dir / filename).resolve()
        if target.parent != self.runs_dir.resolve() or not target.is_file():
            raise FileNotFoundError(filename)
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [row for row in rows if row.get("type") != "request"]


async def event_stream(run: DemoRun) -> AsyncIterator[str]:
    """Every event of the run for this connection alone (its own read position), then a close event."""
    sent = 0
    while True:
        batch, closed = await asyncio.to_thread(run.wait_from, sent)
        for event in batch:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        sent += len(batch)
        if closed and not batch:
            yield "event: close\ndata: {}\n\n"
            return
        if not batch:
            yield ": keepalive\n\n"

"""Thread-safe event storage, SSE delivery, persistence and replay."""
from __future__ import annotations

import asyncio
import json
import queue
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
    queue: queue.Queue[dict[str, Any] | None] = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False

    def publish(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)
        self.queue.put(event)

    def close(self) -> None:
        self.closed = True
        self.queue.put(None)


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
        rows = [{"type": "request", "data": run.request, "run_id": run.run_id, "created_at": run.created_at}, *run.events]
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
    """Send already-published events first, then follow the run until it closes."""
    with run.lock:
        existing = list(run.events)
    for event in existing:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    already_sent = len(existing)
    while True:
        item = await asyncio.to_thread(run.queue.get)
        if item is None:
            yield "event: close\ndata: {}\n\n"
            return
        # The queue contains the same events that were copied above. Only
        # events arriving after the initial snapshot should be streamed here.
        if already_sent:
            already_sent -= 1
            continue
        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

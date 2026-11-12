"""FastAPI server for the local GRASP presentation demo."""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_config, load_dotenv
from .runners import COLUMNS, run_column
from .services import DemoServices
from .store import RunStore, event_stream


class AskRequest(BaseModel):
    eval_id: str | None = None
    question: str | None = None
    corpus: str = "wiki18"
    columns: list[str] = Field(default_factory=lambda: list(COLUMNS))


def _load_eval(config: dict[str, Any]) -> list[dict[str, Any]]:
    from grasp_grpo.data import read_examples

    path = config.get("eval", {}).get("qa")
    if not path or not Path(path).is_file():
        return []
    rows = read_examples(path)
    ids_path = config.get("eval", {}).get("ids")
    if ids_path and Path(ids_path).is_file():
        ids = set(json.loads(Path(ids_path).read_text(encoding="utf-8")).get("ids", []))
        if ids:
            rows = [row for row in rows if row["id"] in ids]
    return rows


def create_app(config_path: str | Path, *, services: DemoServices | None = None) -> FastAPI:
    config = load_config(config_path)
    load_dotenv(config["_root"])
    store = RunStore(config.get("runs_dir", Path(config["_root"]) / "runs"))
    eval_rows = _load_eval(config)
    services = services or DemoServices.from_config(config)
    started = time.monotonic()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        services.close()

    app = FastAPI(title="GRASP Demo", lifespan=lifespan)
    app.state.config = config
    app.state.store = store
    app.state.services = services
    app.state.eval_rows = eval_rows
    # Config copies commonly live beside a user's checkpoints; the web assets
    # live with this package, not necessarily beside that copied YAML file.
    static = Path(__file__).resolve().parents[2] / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        columns = {column: {"enabled": services.enabled(column)[0], "reason": services.enabled(column)[1]} for column in COLUMNS}
        return {"uptime_seconds": time.monotonic() - started, "components": services.health, "columns": columns, "adapters": services.adapter_metadata, "eval_questions": len(eval_rows)}

    @app.get("/api/eval")
    def eval_questions(dataset: str | None = None, query: str = ""):
        normalized = query.casefold().strip()
        rows = [row for row in eval_rows if (not dataset or row.get("metadata", {}).get("dataset") == dataset) and (not normalized or normalized in row["question"].casefold())]
        return [{"id": row["id"], "dataset": row.get("metadata", {}).get("dataset"), "question": row["question"], "gold_answers": row["gold_answers"], "gold_doc_ids": row.get("gold_doc_ids", [])} for row in rows]

    def resolve(request: AskRequest) -> dict[str, Any]:
        if request.eval_id:
            found = next((row for row in eval_rows if row["id"] == request.eval_id), None)
            if found is None:
                raise HTTPException(404, f"unknown eval_id: {request.eval_id}")
            return found
        if not request.question or not request.question.strip():
            raise HTTPException(422, "provide eval_id or a non-empty question")
        return {"id": f"free::{int(time.time() * 1000)}", "question": request.question.strip(), "gold_answers": [], "gold_doc_ids": [], "metadata": {"dataset": request.corpus}}

    def launch(run, example: dict[str, Any], columns: list[str]) -> None:
        begun = time.monotonic()

        def publish(column: str, kind: str, data: dict[str, Any]) -> None:
            run.publish({"run_id": run.run_id, "column": column, "type": kind, "data": data, "t": time.monotonic() - begun})

        def one(column: str) -> None:
            try:
                run_column(services, column, example, lambda kind, data: publish(column, kind, data))
            except Exception as error:  # column failures must not block the four others
                publish(column, "error", {"message": f"{type(error).__name__}: {error}"})

        try:
            with ThreadPoolExecutor(max_workers=len(columns)) as pool:
                futures = [pool.submit(one, column) for column in columns]
                for future in as_completed(futures):
                    future.result()
        finally:
            store.persist(run)
            run.close()

    @app.post("/api/ask")
    def ask(request: AskRequest):
        invalid = sorted(set(request.columns) - set(COLUMNS))
        if invalid:
            raise HTTPException(422, f"unknown columns: {', '.join(invalid)}")
        if not request.columns:
            raise HTTPException(422, "select at least one column")
        example = resolve(request)
        run = store.create({"example": example, "columns": request.columns, "config": str(config.get("_root"))})
        thread = threading.Thread(target=launch, args=(run, example, request.columns), name=f"grasp-demo-{run.run_id}", daemon=True)
        thread.start()
        return {"run_id": run.run_id}

    @app.get("/api/stream/{run_id}")
    async def stream(run_id: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return StreamingResponse(event_stream(run), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/history")
    def history():
        return store.history()

    @app.get("/api/replay/{filename}")
    def replay(filename: str):
        try:
            return JSONResponse(store.replay(filename))
        except FileNotFoundError:
            raise HTTPException(404, "saved run not found") from None

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local GRASP demonstration")
    parser.add_argument("--config", default="models.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.config), host=args.host, port=args.port)

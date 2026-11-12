"""Column runners. They call the trainer's shared baseline and GRASP loops."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from grasp_grpo.baselines import run_ircot, run_single_step
from grasp_grpo.infer import run_example
from grasp_grpo.retrieval import BoundRetriever

from .metrics import score

ColumnEmit = Callable[[str, dict[str, Any]], None]
COLUMNS = ("rag_7b", "rag_235b", "ircot_7b", "grasp_sft", "grasp_grpo")


def _public_docs(data: dict[str, Any], gold: set[str]) -> dict[str, Any]:
    """Do not mutate retriever cache objects while decorating SSE document events."""
    if "docs" not in data:
        return data
    copied = dict(data)
    copied["docs"] = [{key: value for key, value in item.items() if key != "triples"} | {"gold": str(item.get("doc_id")) in gold} for item in data["docs"]]
    return copied


def run_column(services, column: str, example: dict[str, Any], emit: ColumnEmit) -> dict[str, Any]:
    if column not in COLUMNS:
        raise ValueError(f"unknown demo column: {column}")
    enabled, reason = services.enabled(column)
    if not enabled:
        raise RuntimeError(reason or "column is disabled")

    gold = {str(value) for value in example.get("gold_doc_ids", [])}
    started = time.monotonic()
    retrievals = 0

    def traced(kind: str, data: dict[str, Any]) -> None:
        nonlocal retrievals
        if kind == "retrieve":
            retrievals += 1
        emit(kind, _public_docs(data, gold))

    bound = BoundRetriever(services.retriever, example, services.retriever_lock)
    if column == "rag_7b":
        result = run_single_step(services.reader, bound, example, emit=traced)
    elif column == "rag_235b":
        result = run_single_step(services.strong_reader, bound, example, emit=traced)
    elif column == "ircot_7b":
        config = services.config.get("ircot", {})
        result = run_ircot(services.reader, bound, example, max_steps=int(config.get("max_steps", 5)), docs_per_step=int(config.get("docs_per_step", 3)), emit=traced)
    else:
        adapter = column.removeprefix("grasp_")
        grasp = services.config.get("grasp", {}) or {}
        stop = grasp.get("max_empty_ki")  # demo-only stop rule; absent = the evaluated method
        result = run_example(services.proxy.generate_fn(adapter), services.reader, bound, example, int(grasp.get("max_steps", 5)),
                             emit=traced, max_empty_ki=int(stop) if stop else None)

    result["retrievals"] = result.get("retrievals", retrievals)
    metrics = score(example, result)
    emit("done", {"seconds": time.monotonic() - started, "retrievals": result["retrievals"],
                  "context_docs": len(result.get("context_doc_ids", [])), **metrics})
    return result

"""Per-question scores used by both streaming runs and replay."""
from __future__ import annotations

from typing import Any

from grasp_grpo.evaluate import exact, f1


def score(example: dict[str, Any], result: dict[str, Any]) -> dict[str, float | None]:
    if not example.get("gold_answers"):
        return {"f1": None, "em": None, "recall": None}
    gold = {str(value) for value in example.get("gold_doc_ids", [])}
    docs = {str(value) for value in result.get("context_doc_ids", [])}
    return {
        "f1": f1(str(result.get("prediction", "")), list(example["gold_answers"])),
        "em": exact(str(result.get("prediction", "")), list(example["gold_answers"])),
        "recall": len(gold & docs) / len(gold) if gold else None,
    }

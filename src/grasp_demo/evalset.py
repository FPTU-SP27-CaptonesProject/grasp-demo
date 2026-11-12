"""Eval questions shown in the demo: the QA file filtered by an id list (test_ids.json)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ids(path: str | None) -> set[str]:
    if not path or not Path(path).is_file():
        return set()
    return {str(value) for value in json.loads(Path(path).read_text(encoding="utf-8")).get("ids", [])}


def load_eval_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    from grasp_grpo.data import read_examples

    values = (config.get("eval") or {})
    path = values.get("qa")
    if not path or not Path(path).is_file():
        return []
    rows = read_examples(path)
    ids = load_ids(values.get("ids"))
    return [row for row in rows if row["id"] in ids] if ids else rows


def load_gold_titles(config: dict[str, Any]) -> dict[str, str]:
    """{doc_id: title} written by fetch_data next to test_ids.json (gold_titles.json); empty when missing."""
    ids = (config.get("eval") or {}).get("ids")
    path = Path(ids).with_name("gold_titles.json") if ids else None
    if not path or not path.is_file():
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {}

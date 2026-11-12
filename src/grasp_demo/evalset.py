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

"""Showcase questions: GRPO answers exactly (F1 = 1) where every baseline is below 0.5; per dataset, more
gold documents first (multi-hop), then id. Output {"ids": [...], "source": ...} for eval.showcase."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _scores(report: dict[str, Any]) -> dict[str, float]:
    return {str(row["id"]): float(row["f1"]) for row in report["per_question"]}


def pick_showcase(examples: list[dict[str, Any]], grpo: dict[str, Any], baselines: list[dict[str, Any]], *, per_dataset: int = 3) -> list[str]:
    ours, theirs = _scores(grpo), [_scores(report) for report in baselines]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        key = example["id"]
        if ours.get(key) == 1.0 and all(scores.get(key, 0.0) < 0.5 for scores in theirs):
            by_dataset[str(example.get("metadata", {}).get("dataset", ""))].append(example)
    chosen = []
    for dataset in sorted(by_dataset):
        rows = sorted(by_dataset[dataset], key=lambda row: (-len(row.get("gold_doc_ids", [])), row["id"]))
        chosen.extend(row["id"] for row in rows[:per_dataset])
    return chosen


def main(argv: list[str] | None = None) -> None:
    from grasp_grpo.data import read_examples

    parser = argparse.ArgumentParser(description="Pick showcase questions from test999 reports")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--grpo", required=True, help="report of the GRPO checkpoint the demo shows")
    parser.add_argument("--baseline", action="append", required=True, help="baseline report (repeat: single, ircot-k1)")
    parser.add_argument("--per-dataset", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    ids = set(json.loads(Path(args.ids).read_text(encoding="utf-8"))["ids"])
    examples = [row for row in read_examples(args.qa) if row["id"] in ids]
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))  # noqa: E731
    chosen = pick_showcase(examples, load(args.grpo), [load(path) for path in args.baseline], per_dataset=args.per_dataset)
    Path(args.output).write_text(json.dumps({"ids": chosen, "source": Path(args.grpo).name}, indent=2), encoding="utf-8")
    print(json.dumps({"showcase": len(chosen), "output": args.output}))


if __name__ == "__main__":
    main()

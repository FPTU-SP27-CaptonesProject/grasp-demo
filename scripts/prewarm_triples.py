"""Extract triples ahead of time for the documents GRASP retrieves on a question set, without the reader or proxy.

GRASP searches the question first, then every query its QF writes. Saved policy_eval reports keep those queries
(actions[].response), and retrieval is deterministic (same E5, FAISS index and rescore), so re-running exactly
those searches finds the documents the demo will retrieve. Documents missing from the triple cache are extracted
(Bedrock, BEDROCK_API_KEY) into --cache, which can then be uploaded as the demo's cache.

Low memory: one corpus at a time, FAISS index memory-mapped, E5 on the GPU when available.

  python scripts/prewarm_triples.py --reports kg-s250.json kg-grasp-qwen2.5-3b-sft.json \\
      --qa data/grasp-gold-evidence/dev.jsonl --corpus-config data/corpus.json --cache data/grasp-retrieval/cache/triples-demo.sqlite
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def queries_by_corpus(reports: list[dict], examples: dict[str, dict], ids: set[str] | None) -> dict[str, set[str]]:
    """{corpus: {question and every QF query of the reports}} (an invalid QF searched the question itself)."""
    from grasp_grpo.contracts import parse

    wanted: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        for prediction in report["predictions"]:
            example = examples.get(prediction["id"])
            if example is None or (ids is not None and prediction["id"] not in ids):
                continue
            corpus = str(example["metadata"].get("dataset", ""))
            wanted[corpus].add(example["question"])
            for action in prediction.get("actions", []):
                if action.get("role") != "qf":
                    continue
                parsed, valid, _ = parse("qf", action.get("response", ""))
                query = str((parsed or {}).get("query") or example["question"]) if valid else example["question"]
                wanted[corpus].add(query)
    return wanted


def main(argv: list[str] | None = None) -> None:
    import faiss

    from grasp_grpo.data import read_examples
    from grasp_grpo.extractor import TripleExtractor
    from grasp_grpo.retrieval import DenseRetriever, _load_encoder

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports", nargs="+", required=True, help="policy_eval reports with predictions (QF queries)")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--ids", help="optional {\"ids\": [...]} file restricting the questions")
    parser.add_argument("--corpus-config", required=True)
    parser.add_argument("--cache", required=True, help="Project A triple cache (sqlite) to fill in place")
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--device", default=None, help="E5 device (default: cuda if available)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args(argv)

    examples = {row["id"]: row for row in read_examples(args.qa)}
    ids = set(json.loads(Path(args.ids).read_text(encoding="utf-8"))["ids"]) if args.ids else None
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    wanted = queries_by_corpus(reports, examples, ids)
    config = json.loads(Path(args.corpus_config).read_text(encoding="utf-8"))
    encoder = _load_encoder(args.encoder, args.device)
    extractor = TripleExtractor(str(Path(args.cache).resolve()), budget_usd=args.budget_usd, concurrency=args.concurrency)
    summary = {}
    for corpus, queries in sorted(wanted.items()):
        root = Path(config[corpus]["index_dir"])
        path = str(root / args.encoder.rsplit("/", 1)[-1] / "index.faiss")
        try:
            index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
        except RuntimeError:  # some builds (Windows) cannot memory-map IVF inverted lists: load it normally
            index = faiss.read_index(path)
        retriever = DenseRetriever.from_parts(index, encoder, root / "passages.sqlite", top_k=args.top_k, triple_cache=args.cache)
        retriever.extractor = extractor
        # Search everything first (fill=False only flags documents missing from the cache), then extract all of
        # them in one parallel batch: filling per search would keep at most top_k Bedrock calls in flight.
        missing: dict[str, dict] = {}
        for query in sorted(queries):
            for item in retriever.search(query, fill=False):
                if item.get("needs_triples"):
                    missing.setdefault(item["doc_id"], item)
        print(json.dumps({corpus: {"queries": len(queries), "documents_to_extract": len(missing)}}), flush=True)
        extractor.fill(list(missing.values()))
        retriever.extractor = None  # shared; closed once below
        retriever.close()
        summary[corpus] = {"queries": len(queries), **retriever.stats}
        print(json.dumps({corpus: summary[corpus]}), flush=True)
    extractor.close()
    print(json.dumps({"extraction": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in extractor.stats.items()}}))


if __name__ == "__main__":
    main()

"""A merged "all" corpus for free questions: search every corpus, keep the best top_k by E5 score.

All corpora share the E5 encoder, so their (rescored) cosine scores are comparable. Eval questions keep
their own corpus (as in the reported evals); only free questions sent with corpus "all" use this.
"""
from __future__ import annotations

from typing import Any

ALL = "all"


class AllCorpora:
    def __init__(self, router, top_k: int):
        self.router, self.top_k = router, top_k
        self.extractor = None  # never extracts: BoundRetriever then calls search() directly

    def search(self, query: str, *, fill: bool = True) -> list[dict[str, Any]]:
        merged = []
        for name, retriever in self.router.retrievers.items():
            for item in retriever.search(query, fill=fill):
                merged.append({**item, "corpus": name})
        merged.sort(key=lambda item: -float(item.get("score") or 0.0))
        kept, seen = [], set()
        for item in merged:
            key = " ".join(str(item.get("text", "")).split())[:200].casefold() or item["doc_id"]
            if key in seen:  # the same passage in two Wikipedia-based corpora
                continue
            seen.add(key)
            kept.append(item)
            if len(kept) >= self.top_k:
                break
        return kept

    def candidate_triples(self, query: str, passages: list[dict[str, Any]], question: str | None = None) -> list[dict[str, Any]]:
        # Triples are already attached to each passage by its own corpus; ranking only needs the shared encoder.
        return next(iter(self.router.retrievers.values())).candidate_triples(query, passages, question=question)

    def record(self, result: list[dict[str, Any]]) -> None:
        return None


class RouterWithAll:
    """grasp_grpo CorpusRouter plus the virtual corpus "all"."""

    def __init__(self, router, top_k: int):
        self.router, self.all = router, AllCorpora(router, top_k)
        self.retrievers = router.retrievers
        self.current = None

    @property
    def stats(self):
        return self.router.stats

    def select(self, example: dict[str, Any]) -> None:
        if str((example.get("metadata") or {}).get("dataset", "")) == ALL:
            self.current = self.all
        else:
            self.router.select(example)
            self.current = self.router.current

    def search(self, query: str, *, fill: bool = True) -> list[dict[str, Any]]:
        if self.current is None:
            raise RuntimeError("select(example) must be called before retrieval")
        return self.current.search(query, fill=fill)

    def candidate_triples(self, query: str, passages: list[dict[str, Any]], question: str | None = None) -> list[dict[str, Any]]:
        if self.current is None:
            raise RuntimeError("select(example) must be called before retrieval")
        return self.current.candidate_triples(query, passages, question=question)

    def close(self) -> None:
        self.router.close()

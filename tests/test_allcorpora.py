from grasp_demo.allcorpora import RouterWithAll


class Dense:
    def __init__(self, docs):
        self.docs, self.extractor, self.current = docs, None, None

    def search(self, query, *, fill=True):
        return [dict(d) for d in self.docs]

    def candidate_triples(self, query, passages, question=None):
        return [t for p in passages for t in p.get("triples", [])]


class Router:
    def __init__(self, retrievers):
        self.retrievers, self.current, self.closed = retrievers, None, False

    @property
    def stats(self):
        return {}

    def select(self, example):
        self.current = self.retrievers[example["metadata"]["dataset"]]

    def close(self):
        self.closed = True


def make():
    wiki = Dense([{"doc_id": "w1", "title": "Up", "text": "Up is a film", "score": 0.80, "triples": [{"doc_id": "w1", "head": "Up", "relation": "r", "tail": "t"}]},
                  {"doc_id": "w2", "title": "Up", "text": "Second chunk", "score": 0.60, "triples": []}])
    hotpot = Dense([{"doc_id": "h1", "title": "Up (film)", "text": "Up  is a  film", "score": 0.79, "triples": []},
                    {"doc_id": "h2", "title": "Pete Docter", "text": "Docter directed Up", "score": 0.75, "triples": []}])
    return RouterWithAll(Router({"wiki18": wiki, "hotpotqa_abstracts_2017": hotpot}), top_k=3)


def test_all_merges_by_score_drops_duplicate_passages_and_keeps_top_k():
    router = make()
    router.select({"metadata": {"dataset": "all"}})
    docs = router.search("who directed up")
    assert [(d["doc_id"], d["corpus"]) for d in docs] == [("w1", "wiki18"), ("h2", "hotpotqa_abstracts_2017"), ("w2", "wiki18")]
    assert router.candidate_triples("q", docs)[0]["head"] == "Up"


def test_a_named_dataset_still_uses_its_own_corpus():
    router = make()
    router.select({"metadata": {"dataset": "hotpotqa_abstracts_2017"}})
    assert [d["doc_id"] for d in router.search("q")] == ["h1", "h2"]
    router.close()
    assert router.router.closed

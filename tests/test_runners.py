from grasp_demo.runners import run_column


class DummyServices:
    def __init__(self):
        self.health = {"reader": {"ready": True}, "retriever": {"ready": True}, "proxy": {"ready": True}, "strong": {"ready": False, "reason": "no key"}}
        self.config = {"ircot": {}, "grasp": {}}

    def enabled(self, column):
        return (False, "no key") if column == "rag_235b" else (True, None)


def test_disabled_column_is_rejected_without_starting_a_runner():
    try:
        run_column(DummyServices(), "rag_235b", {"id": "q", "question": "Q", "metadata": {}}, lambda *_: None)
    except RuntimeError as error:
        assert "no key" in str(error)
    else:
        raise AssertionError("expected disabled column")


import threading

from grasp_grpo.evaluate import f1
from grasp_grpo.model import Action
from grasp_grpo.reader import ReaderStep

EXAMPLE = {"id": "q", "question": "Q?", "gold_answers": ["A"], "gold_doc_ids": ["d2"], "metadata": {"dataset": "wiki18"}}


class Retriever:
    def search(self, query):
        return [{"doc_id": "d2" if query == "gold query" else "d1", "title": "T", "text": "x", "triples": [{"h": 1}]}]

    def candidate_triples(self, query, passages, question=None):
        return [{"doc_id": p["doc_id"], "head": "H", "relation": "r", "tail": "T"} for p in passages]


class Reader:
    def __init__(self):
        self.calls = 0

    def step(self, question, context, thoughts, transcript):
        self.calls += 1
        return ReaderStep("<think> need </think>\n", "need", None) if self.calls == 1 else ReaderStep("<answer> A </answer>", None, "A")

    def final(self, *args):
        return "A"


class Proxy:
    names = {"sft": "default", "grpo": "grpo"}

    def generate_fn(self, adapter):
        texts = {"rd": "Yes (retrieval is required)", "qf": "Reasoning Chain: None\nKnowledge Gap: need\nQuery: gold query", "ki": "<H; r; T>"}
        return lambda role, prompt, samples: [Action(texts[role], [1], [0.0])]


class Services:
    config = {"ircot": {}, "grasp": {}}

    def __init__(self):
        self.retriever, self.retriever_lock, self.reader, self.proxy = Retriever(), threading.Lock(), Reader(), Proxy()

    def enabled(self, column):
        return True, None


def test_grasp_column_streams_steps_and_scores():
    events = []
    result = run_column(Services(), "grasp_grpo", EXAMPLE, lambda kind, data: events.append((kind, data)))
    kinds = [kind for kind, _ in events]
    assert kinds == ["retrieve", "thought", "rd", "qf", "retrieve", "ki", "answer", "done"]
    first_docs = events[0][1]["docs"]
    assert "triples" not in first_docs[0] and first_docs[0]["gold"] is False
    assert events[4][1]["docs"][0]["gold"] is True
    done = events[-1][1]
    assert done["f1"] == f1(result["prediction"], ["A"]) == 1.0 and done["recall"] == 1.0 and done["retrievals"] == 2
    assert done["context_docs"] == len(result["context_doc_ids"]) == 2



class NoTripleRetriever(Retriever):
    def candidate_triples(self, query, passages, question=None):
        return []


class StopServices(Services):
    config = {"ircot": {}, "grasp": {"max_empty_ki": 1}}

    def __init__(self):
        super().__init__()
        self.retriever = NoTripleRetriever()


def test_grasp_stop_rule_from_config():
    events = []
    result = run_column(StopServices(), "grasp_grpo", EXAMPLE, lambda kind, data: events.append((kind, data)))
    assert result["prediction"] == "Insufficient information"
    answer = next(d for k, d in events if k == "answer")
    assert "no triple" in answer["reason"]

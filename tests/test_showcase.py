from grasp_demo.showcase import pick_showcase


def report(scores):
    return {"per_question": [{"id": k, "f1": v, "em": v} for k, v in scores.items()]}


def test_pick_prefers_grpo_wins_with_more_gold_docs_per_dataset():
    examples = [
        {"id": "a", "gold_doc_ids": ["1"], "metadata": {"dataset": "wiki18"}},
        {"id": "b", "gold_doc_ids": ["1", "2"], "metadata": {"dataset": "wiki18"}},
        {"id": "c", "gold_doc_ids": ["1", "2"], "metadata": {"dataset": "2wiki_ircot"}},
        {"id": "d", "gold_doc_ids": ["1", "2"], "metadata": {"dataset": "2wiki_ircot"}},
        {"id": "e", "gold_doc_ids": ["1"], "metadata": {"dataset": "wiki18"}},
    ]
    grpo = report({"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.5, "e": 1.0})
    single = report({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0})
    ircot = report({"a": 0.2, "b": 0.0, "c": 0.4, "d": 0.0, "e": 0.9})
    assert pick_showcase(examples, grpo, [single, ircot], per_dataset=1) == ["c", "b"]
    assert pick_showcase(examples, grpo, [single, ircot], per_dataset=3) == ["c", "b", "a"]

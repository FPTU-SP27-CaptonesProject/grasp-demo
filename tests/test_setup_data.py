import json
import sqlite3

import pytest

from grasp_demo.setup_data import CORPORA, check_eval_files, corpus_config, fetch_adapter


def make_tree(root, *, skip=None, triples_table=True):
    for corpus in CORPORA:
        (root / corpus / "e5-large-v2").mkdir(parents=True)
        (root / corpus / "passages.sqlite").write_bytes(b"")
        if corpus != skip:
            (root / corpus / "e5-large-v2" / "index.faiss").write_bytes(b"")
    (root / "cache").mkdir()
    db = sqlite3.connect(root / "cache" / "t.sqlite")
    if triples_table:
        db.execute("CREATE TABLE triples (doc_id TEXT, model TEXT, triples TEXT)")
    db.commit()
    db.close()
    return root / "cache" / "t.sqlite"


def test_corpus_config_points_every_corpus_at_one_triple_cache(tmp_path):
    triples = make_tree(tmp_path)
    config = corpus_config(tmp_path, triples)
    assert set(config) == set(CORPORA)
    assert config["wiki18"] == {"index_dir": str((tmp_path / "wiki18").resolve()), "triple_cache": str(triples.resolve())}


def test_corpus_config_names_the_missing_index(tmp_path):
    triples = make_tree(tmp_path, skip="wiki18")
    with pytest.raises(FileNotFoundError, match=r"wiki18.e5-large-v2.index\.faiss"):
        corpus_config(tmp_path, triples)


def test_corpus_config_rejects_a_cache_without_triples_table(tmp_path):
    triples = make_tree(tmp_path, triples_table=False)
    with pytest.raises(FileNotFoundError, match="no triples table"):
        corpus_config(tmp_path, triples)


def test_check_eval_files_reports_missing_ids(tmp_path):
    qa = tmp_path / "dev.jsonl"
    qa.write_text(json.dumps({"id": "a", "question": "Q?", "gold_answers": ["x"]}) + "\n", encoding="utf-8")
    ids = tmp_path / "test_ids.json"
    ids.write_text(json.dumps({"ids": ["a"]}), encoding="utf-8")
    assert check_eval_files(qa, ids) == 1
    ids.write_text(json.dumps({"ids": ["a", "b"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="b"):
        check_eval_files(qa, ids)


def test_fetch_adapter_lifts_a_subfolder_to_the_top(tmp_path):
    def download(repo, *, repo_type, revision, local_dir, allow_patterns):
        assert allow_patterns == ["s100/*"]
        (local_dir / "s100").mkdir(parents=True)
        (local_dir / "s100" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (local_dir / "s100" / "adapter_model.safetensors").write_bytes(b"w")

    target = fetch_adapter("org/ckpts", "grpo-s100", tmp_path, subfolder="s100", download=download)
    assert target == tmp_path / "grpo-s100"
    assert sorted(p.name for p in target.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    assert not (tmp_path / ".staging").exists()


def test_build_gold_titles_reads_titles_for_the_question_set(tmp_path):
    from grasp_demo.setup_data import build_gold_titles

    corpus = tmp_path / "wiki18"
    corpus.mkdir()
    db = sqlite3.connect(corpus / "passages.sqlite")
    db.execute('CREATE TABLE passages ("row" INTEGER, doc_id TEXT, title TEXT, text TEXT)')
    db.executemany("INSERT INTO passages VALUES (?, ?, ?, ?)", [(0, "d1", "Up (2009 film)", "x"), (1, "d2", "Pete Docter", "y"), (2, "d3", "Other", "z")])
    db.commit()
    db.close()
    qa = tmp_path / "dev.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "question": "Q?", "gold_answers": ["x"], "gold_doc_ids": ["d1", "d2", "missing"], "dataset": "wiki18"},
        {"id": "b", "question": "Q2?", "gold_answers": ["x"], "gold_doc_ids": ["d3"], "dataset": "wiki18"}]), encoding="utf-8")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"ids": ["a"]}), encoding="utf-8")
    titles = build_gold_titles(qa, ids, {"wiki18": {"index_dir": str(corpus), "triple_cache": ""}})
    assert titles == {"d1": "Up (2009 film)", "d2": "Pete Docter"}


def test_prewarm_collects_the_question_and_every_qf_query_per_corpus():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("prewarm", Path(__file__).resolve().parents[1] / "scripts" / "prewarm_triples.py")
    prewarm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prewarm)
    examples = {"a": {"id": "a", "question": "Q a?", "metadata": {"dataset": "wiki18"}},
                "b": {"id": "b", "question": "Q b?", "metadata": {"dataset": "2wiki_ircot"}}}
    report = {"predictions": [
        {"id": "a", "actions": [{"role": "rd", "response": "Yes"},
                                {"role": "qf", "response": "Reasoning Chain: None\nKnowledge Gap: who\nQuery: director of Up"},
                                {"role": "qf", "response": "garbage"}]},
        {"id": "b", "actions": []}]}
    wanted = prewarm.queries_by_corpus([report], examples, None)
    assert wanted == {"wiki18": {"Q a?", "director of Up"}, "2wiki_ircot": {"Q b?"}}
    assert prewarm.queries_by_corpus([report], examples, {"b"}) == {"2wiki_ircot": {"Q b?"}}

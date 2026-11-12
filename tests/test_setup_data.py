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

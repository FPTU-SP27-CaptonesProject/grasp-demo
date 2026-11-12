import json

from grasp_demo.config import load_config, reader_api_key
from grasp_demo.evalset import load_eval_rows, load_ids


def write(tmp_path, text):
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_adapters_accept_strings_and_entries(tmp_path):
    config = load_config(write(tmp_path, "proxy:\n  adapters:\n    sft: ck/sft\n    grpo: {path: ck/s100, label: GRPO s100}\n"))
    adapters = config["proxy"]["adapters"]
    assert adapters["sft"] == {"path": str(tmp_path / "ck" / "sft"), "label": "sft"}
    assert adapters["grpo"] == {"path": str(tmp_path / "ck" / "s100"), "label": "GRPO s100"}


def test_eval_paths_are_resolved(tmp_path):
    config = load_config(write(tmp_path, "eval: {qa: d/dev.jsonl, ids: d/test_ids.json, showcase: d/showcase.json}\nruns_dir: runs\n"))
    assert config["eval"]["showcase"] == str(tmp_path / "d" / "showcase.json")
    assert config["runs_dir"] == str(tmp_path / "runs")


def test_reader_api_key_prefers_named_env(monkeypatch):
    monkeypatch.setenv("DEMO_KEY", "secret")
    assert reader_api_key({"api_key_env": "DEMO_KEY", "api_key": "x"}) == "secret"
    monkeypatch.delenv("DEMO_KEY")
    assert reader_api_key({"api_key_env": "DEMO_KEY"}) == "EMPTY"
    assert reader_api_key({"api_key": "x"}) == "x"


def test_eval_rows_are_filtered_by_ids(tmp_path):
    qa = tmp_path / "dev.jsonl"
    qa.write_text("\n".join(json.dumps({"id": i, "question": f"Q{i}?", "gold_answers": ["a"], "dataset": "wiki18"}) for i in ("a", "b", "c")), encoding="utf-8")
    (tmp_path / "ids.json").write_text(json.dumps({"ids": ["a", "c", "zzz"]}), encoding="utf-8")
    config = load_config(write(tmp_path, "eval: {qa: dev.jsonl, ids: ids.json}\n"))
    assert [row["id"] for row in load_eval_rows(config)] == ["a", "c"]
    assert load_ids(None) == set()
    assert load_ids(str(tmp_path / "missing.json")) == set()


def test_empty_yaml_sections_become_empty_mappings(tmp_path):
    config = load_config(write(tmp_path, "eval:\nproxy:\nretrieval:\nreader:\n"))
    assert config["eval"] == {} and config["proxy"] == {} and config["retrieval"] == {} and config["reader"] == {}

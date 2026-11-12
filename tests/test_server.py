import json

from fastapi.testclient import TestClient

from grasp_demo.server import create_app


class Services:
    def __init__(self):
        self.health = {"reader": {"ready": True}, "retriever": {"ready": True}, "proxy": {"ready": True}, "strong": {"ready": True}}

    def enabled(self, column):
        return True, None

    def close(self):
        pass


def make_config(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({"id": "q1", "question": "Question?", "gold_answers": ["answer"], "gold_doc_ids": ["doc"], "dataset": "wiki18"}) + "\n", encoding="utf-8")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"ids": ["q1"]}), encoding="utf-8")
    config = tmp_path / "models.yaml"
    config.write_text(f"eval:\n  qa: {qa.name}\n  ids: {ids.name}\nruns_dir: runs\n", encoding="utf-8")
    return config


def test_ask_streams_events_and_persists_a_run(tmp_path, monkeypatch):
    def fake_run(services, column, example, emit):
        emit("retrieve", {"query": example["question"], "docs": []})
        emit("answer", {"text": "answer"})
        emit("done", {"seconds": 0.1, "retrievals": 1, "f1": 1.0, "em": 1.0, "recall": 0.0})

    monkeypatch.setattr("grasp_demo.server.run_column", fake_run)
    app = create_app(make_config(tmp_path), services=Services())
    with TestClient(app) as client:
        run_id = client.post("/api/ask", json={"eval_id": "q1", "columns": ["rag_7b", "grasp_sft"]}).json()["run_id"]
        response = client.get(f"/api/stream/{run_id}")
        assert response.status_code == 200
        assert response.text.count('"type": "done"') == 2
        history = client.get("/api/history").json()
        assert len(history) == 1


def test_free_question_has_no_scores(tmp_path, monkeypatch):
    def fake_run(services, column, example, emit):
        emit("done", {"seconds": 0.1, "retrievals": 1, "f1": None, "em": None, "recall": None})

    monkeypatch.setattr("grasp_demo.server.run_column", fake_run)
    app = create_app(make_config(tmp_path), services=Services())
    with TestClient(app) as client:
        response = client.post("/api/ask", json={"question": "Free?", "corpus": "wiki18", "columns": ["rag_7b"]})
        assert response.status_code == 200
        assert client.get("/api/eval").json()[0]["id"] == "q1"

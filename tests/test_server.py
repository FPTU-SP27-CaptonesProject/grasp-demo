import json
import threading

from fastapi.testclient import TestClient

from grasp_demo.server import create_app


class Services:
    adapter_metadata = {"sft": {"label": "SFT", "path": "x"}}

    def __init__(self):
        self.health = {"reader": {"ready": True}, "retriever": {"ready": True}, "proxy": {"ready": True}, "strong": {"ready": True}}

    def enabled(self, column):
        return True, None

    def close(self):
        pass


def make_config(tmp_path):
    qa = tmp_path / "qa.jsonl"
    rows = [{"id": i, "question": f"Question {i}?", "gold_answers": ["answer"], "gold_doc_ids": ["doc"], "dataset": "wiki18"} for i in ("q1", "q2")]
    qa.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (tmp_path / "ids.json").write_text(json.dumps({"ids": ["q1", "q2"]}), encoding="utf-8")
    (tmp_path / "showcase.json").write_text(json.dumps({"ids": ["q2"]}), encoding="utf-8")
    config = tmp_path / "models.yaml"
    config.write_text("eval:\n  qa: qa.jsonl\n  ids: ids.json\n  showcase: showcase.json\nruns_dir: runs\n", encoding="utf-8")
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


def test_eval_lists_showcase_questions(tmp_path):
    app = create_app(make_config(tmp_path), services=Services())
    with TestClient(app) as client:
        assert [row["id"] for row in client.get("/api/eval").json()] == ["q1", "q2"]
        assert [row["id"] for row in client.get("/api/eval?showcase=1").json()] == ["q2"]
        assert client.get("/api/health").json()["showcase_questions"] == 1


def test_prerun_refuses_a_second_concurrent_prerun(tmp_path, monkeypatch):
    release = threading.Event()

    def slow_run(services, column, example, emit):
        release.wait(5)
        emit("done", {"seconds": 0.1, "retrievals": 0, "f1": 1.0, "em": 1.0, "recall": 1.0})

    monkeypatch.setattr("grasp_demo.server.run_column", slow_run)
    app = create_app(make_config(tmp_path), services=Services())
    with TestClient(app) as client:
        first = client.post("/api/prerun")
        assert first.status_code == 200 and first.json()["total"] == 1
        assert client.post("/api/prerun").status_code == 409
        release.set()
        for _ in range(50):
            if not client.get("/api/prerun").json()["running"]:
                break
            threading.Event().wait(0.1)
        status = client.get("/api/prerun").json()
        assert status == {"running": False, "done": 1, "total": 1, "error": None}
        assert len(client.get("/api/history").json()) == 1


def test_stream_still_closes_when_persist_fails(tmp_path, monkeypatch):
    def fake_run(services, column, example, emit):
        emit("done", {"seconds": 0.1, "retrievals": 1, "f1": None, "em": None, "recall": None})

    def boom(self, run):
        raise OSError("disk full")

    monkeypatch.setattr("grasp_demo.server.run_column", fake_run)
    monkeypatch.setattr("grasp_demo.store.RunStore.persist", boom)
    app = create_app(make_config(tmp_path), services=Services())
    with TestClient(app) as client:
        run_id = client.post("/api/ask", json={"question": "Free?", "columns": ["rag_7b"]}).json()["run_id"]
        response = client.get(f"/api/stream/{run_id}")
        assert "event: close" in response.text

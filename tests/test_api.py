from fastapi.testclient import TestClient

from controller.api import create_app
from controller.config import RunConfig, create_controller
from dataclasses import replace


def test_http_lifecycle_and_whitelist(tmp_path):
    service = create_controller(replace(RunConfig(), state_dir=str(tmp_path)))
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json()["engine"]["engine_kind"] == "mock"
        assert client.get("/sessions/not-created").status_code == 404
        payload = {"query": "timeout?", "request_id": "r1", "evidence": [
            {"document_id": "a", "text": "timeout=30", "source_uri": "docs://a"}]}
        first = client.post("/sessions/s/turns", json=payload)
        assert first.status_code == 200
        assert first.json()["action"] == "REBUILD"
        assert client.post("/sessions/s/turns", json=payload).json()["idempotent_replay"]
        payload["targets"] = "must never enter serving"
        assert client.post("/sessions/s/turns", json=payload).status_code == 422
        state = client.get("/sessions/s").json()
        assert len(state["raw_history"]) == 1
        assert client.post("/sessions/s/close").status_code == 200
        assert client.post("/sessions/s/turns", json={"query": "q2", "request_id": "r2"}).status_code == 409

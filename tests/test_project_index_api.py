from fastapi.testclient import TestClient

from backend import api
from backend.api import app
from backend.embeddings import EmbeddingError, EmbeddingUnavailableError
from backend.project_indexer import ChromaIndexError


def _fake_project(monkeypatch):
    monkeypatch.setattr(api, "register_project", lambda _path: {"id": "sample", "name": "sample"})


def test_indexing_maps_ollama_timeout_to_actionable_structured_error(monkeypatch, tmp_path):
    _fake_project(monkeypatch)
    monkeypatch.setattr(api, "index_registered_project", lambda _project_id: (_ for _ in ()).throw(EmbeddingUnavailableError("Ollama timed out after automatic retries.")))

    response = TestClient(app).post("/projects/index", json={"project_path": str(tmp_path)})

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "OLLAMA_TIMEOUT",
        "message": "Ollama timed out after automatic retries.",
    }
    assert "Traceback" not in response.text


def test_indexing_maps_invalid_vectors_and_chroma_failures(monkeypatch, tmp_path):
    _fake_project(monkeypatch)
    client = TestClient(app)
    monkeypatch.setattr(api, "index_registered_project", lambda _project_id: (_ for _ in ()).throw(EmbeddingError("Expected 768 dimensions.")))
    embedding_failure = client.post("/projects/index", json={"project_path": str(tmp_path)})
    assert embedding_failure.status_code == 502
    assert embedding_failure.json()["detail"]["code"] == "EMBEDDING_FAILED"

    monkeypatch.setattr(api, "index_registered_project", lambda _project_id: (_ for _ in ()).throw(ChromaIndexError("Vector store unavailable.")))
    chroma_failure = client.post("/projects/index", json={"project_path": str(tmp_path)})
    assert chroma_failure.status_code == 500
    assert chroma_failure.json()["detail"]["code"] == "CHROMA_FAILED"


def test_invalid_project_path_has_actionable_error_code(tmp_path):
    invalid = tmp_path / "does-not-exist"
    response = TestClient(app).post("/projects/index", json={"project_path": str(invalid)})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "PROJECT_PATH_INVALID"

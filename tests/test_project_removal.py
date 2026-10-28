from fastapi.testclient import TestClient
from types import SimpleNamespace

from backend import project_manager
from backend.api import app
from backend.project_manager import register_project
from backend.vector_store import VectorStore
from backend import vector_store


def _registered(tmp_path, monkeypatch, *project_names):
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", tmp_path / "projects.json")
    result = []
    for name in project_names:
        root = tmp_path / name
        root.mkdir()
        (root / "keep.txt").write_text("source remains", encoding="utf-8")
        result.append((root, register_project(root)))
    return result


def test_remove_project_deletes_index_metadata_but_preserves_source_and_other_projects(tmp_path, monkeypatch):
    (first_root, first), (_, second) = _registered(tmp_path, monkeypatch, "first", "second")
    removed_collections = []
    monkeypatch.setattr(VectorStore, "delete_project_collection", lambda project_id: removed_collections.append(project_id) or True)

    response = TestClient(app).delete(f"/projects/{first['id']}")

    assert response.status_code == 200
    assert response.json() == {
        "removed": True, "project_id": first["id"], "name": "first", "source_files_preserved": True,
    }
    assert (first_root / "keep.txt").read_text(encoding="utf-8") == "source remains"
    assert project_manager.get_project(first["id"]) is None
    assert project_manager.get_project(second["id"]) is not None
    assert [item["id"] for item in project_manager.list_projects()] == [second["id"]]
    assert removed_collections == [first["id"]]
    assert TestClient(app).get("/projects").json()["projects"][0]["id"] == second["id"]


def test_remove_project_succeeds_when_its_vector_collection_is_missing(tmp_path, monkeypatch):
    (root, project), = _registered(tmp_path, monkeypatch, "empty-index")
    # False means the isolated collection was already absent; removal is still safe.
    monkeypatch.setattr(VectorStore, "delete_project_collection", lambda _project_id: False)

    response = TestClient(app).delete(f"/projects/{project['id']}")

    assert response.status_code == 200
    assert project_manager.get_project(project["id"]) is None
    assert (root / "keep.txt").read_text(encoding="utf-8") == "source remains"


def test_remove_nonexistent_project_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", tmp_path / "projects.json")
    response = TestClient(app).delete("/projects/missing")
    assert response.status_code == 404


def test_vector_collection_removal_treats_missing_collection_as_successful_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "settings", SimpleNamespace(chroma_path=str(tmp_path / "chroma")))
    assert VectorStore.delete_project_collection("no_collection") is False
    VectorStore(project_id="remove_me")
    assert VectorStore.delete_project_collection("remove_me") is True
    assert VectorStore.delete_project_collection("remove_me") is False

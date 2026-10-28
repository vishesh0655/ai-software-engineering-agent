import pytest

from backend import project_indexer, project_manager
from backend.embeddings import EmbeddingUnavailableError
from backend.project_indexer import EmptyProjectError, index_registered_project


class _MemoryStore:
    def __init__(self):
        self.documents = {}

    def add_documents(self, documents, embeddings, metadatas, ids):
        for document_id, document, vector, metadata in zip(ids, documents, embeddings, metadatas):
            self.documents[document_id] = (document, vector, metadata)

    def remove_documents_except(self, retained_ids):
        retained = set(retained_ids)
        self.documents = {key: value for key, value in self.documents.items() if key in retained}

    def count(self):
        return len(self.documents)


@pytest.fixture
def indexed_project(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", tmp_path / "projects.json")
    project_root = tmp_path / "source"
    project_root.mkdir()
    source = project_root / "sample.py"
    source.write_text("def first():\n    return 1\n\ndef second():\n    return 2\n", encoding="utf-8")
    project = project_manager.register_project(project_root)
    store = _MemoryStore()
    monkeypatch.setattr(project_indexer, "VectorStore", lambda **_kwargs: store)
    monkeypatch.setattr(project_indexer, "settings", type("Settings", (), {"embedding_batch_size": 1})())

    def embed(documents, progress_callback=None):
        total = len(documents)
        if progress_callback:
            for completed in range(1, total + 1):
                progress_callback(completed, total)
        return [[float(index)] * 768 for index, _ in enumerate(documents)]

    monkeypatch.setattr(project_indexer, "get_embeddings", embed)
    return project, source, store


def test_small_project_index_tracks_progress_and_vector_count(indexed_project):
    project, _, store = indexed_project

    result = index_registered_project(project["id"])

    assert result["status"] == "indexed"
    assert result["files_indexed"] == 1
    assert result["chunks_indexed"] == 2
    assert result["vectors_stored"] == store.count() == 2
    assert result["indexing_progress"]["stage"] == "Completed"
    assert result["indexing_progress"]["embedding_batches_completed"] == 2
    assert result["indexing_progress"]["total_embedding_batches"] == 2
    assert result["indexing_progress"]["files_processed"] == 1


def test_reindexing_is_deterministic_and_removes_stale_vectors(indexed_project):
    project, source, store = indexed_project
    index_registered_project(project["id"])
    first_ids = set(store.documents)

    index_registered_project(project["id"])
    assert set(store.documents) == first_ids
    assert store.count() == 2

    source.write_text("def only():\n    return 3\n", encoding="utf-8")
    result = index_registered_project(project["id"])
    assert result["chunk_count"] == result["vectors_stored"] == 1
    assert store.count() == 1
    assert set(store.documents).isdisjoint(first_ids)


def test_embedding_failure_preserves_previous_index_and_counts(indexed_project, monkeypatch):
    project, source, store = indexed_project
    index_registered_project(project["id"])
    previous_ids = set(store.documents)
    source.write_text("def replacement():\n    return 9\n", encoding="utf-8")
    monkeypatch.setattr(project_indexer, "get_embeddings", lambda *_args, **_kwargs: (_ for _ in ()).throw(EmbeddingUnavailableError("Ollama timed out")))

    with pytest.raises(EmbeddingUnavailableError, match="timed out"):
        index_registered_project(project["id"])

    current = project_manager.get_project(project["id"])
    assert current["status"] == "failed"
    assert current["file_count"] == 1
    assert current["chunk_count"] == 2
    assert "timed out" in current["indexing_error"]
    assert set(store.documents) == previous_ids


def test_empty_reindex_does_not_discard_a_previous_valid_index(indexed_project):
    project, source, store = indexed_project
    index_registered_project(project["id"])
    previous_ids = set(store.documents)
    source.write_text("", encoding="utf-8")

    with pytest.raises(EmptyProjectError):
        index_registered_project(project["id"])

    current = project_manager.get_project(project["id"])
    assert current["status"] == "failed"
    assert current["chunk_count"] == 2
    assert set(store.documents) == previous_ids


def test_new_empty_project_is_reported_as_empty(indexed_project):
    project, source, _ = indexed_project
    source.write_text("", encoding="utf-8")

    with pytest.raises(EmptyProjectError):
        index_registered_project(project["id"])

    current = project_manager.get_project(project["id"])
    assert current["status"] == "empty"
    assert current["chunk_count"] == 0

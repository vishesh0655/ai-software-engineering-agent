from typing import Any
import logging
import time

from backend.chunking import chunk_file
from backend.embeddings import EmbeddingError, EmbeddingUnavailableError, get_embeddings
from backend.ingestion import get_project_files
from backend.indexer import create_chunk_id
from backend.project_manager import (
    ProjectNotFoundError,
    get_project,
    normalize_project_path,
    update_project,
)
from backend.settings import settings
from backend.vector_store import VectorStore

logger = logging.getLogger("backend.project_indexer")


class EmptyProjectError(ValueError):
    pass


class ChromaIndexError(RuntimeError):
    """The local vector database could not store the completed index."""


def index_registered_project(project_id: str) -> dict[str, Any]:
    started = time.monotonic()
    project = get_project(project_id)
    if project is None:
        raise ProjectNotFoundError(f"Project '{project_id}' was not found")

    path = normalize_project_path(project["path"])
    previous_counts = {key: project.get(key, 0) for key in ("file_count", "chunk_count")}
    def progress(stage: str, **values: Any) -> None:
        update_project(project_id, status="indexing", indexing_error=None, indexing_progress={"stage": stage, **values})
    try:
        progress("Scanning files", files_discovered=0, files_processed=0, chunks_created=0, embedding_batches_completed=0, total_embedding_batches=0, vectors_stored=0)
        files = get_project_files(path)
        progress("Chunking code", files_discovered=len(files), files_processed=0, chunks_created=0, embedding_batches_completed=0, total_embedding_batches=0, vectors_stored=0)
        chunks = []
        for index, file_path in enumerate(files, start=1):
            try:
                chunks.extend(chunk_file(file_path))
            except PermissionError:
                logger.warning("Skipping unreadable project file during chunking: %s", file_path)
            if index == 1 or index % 10 == 0 or index == len(files):
                progress("Chunking code", files_discovered=len(files), files_processed=index, chunks_created=len(chunks), embedding_batches_completed=0, total_embedding_batches=0, vectors_stored=0)
        indexed_files = {chunk["metadata"]["file"] for chunk in chunks}
        file_count = len(indexed_files)
        if not chunks:
            raise EmptyProjectError("The project contains no supported, non-empty code chunks")
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [create_chunk_id(chunk) for chunk in chunks]
        progress("Generating embeddings", files_discovered=len(files), files_processed=len(files), chunks_created=len(chunks), embedding_batches_completed=0, total_embedding_batches=(len(documents) + settings.embedding_batch_size - 1) // settings.embedding_batch_size, vectors_stored=0)
        def embedding_progress(completed: int, total: int) -> None:
            progress("Generating embeddings", files_discovered=len(files), files_processed=len(files), chunks_created=len(chunks), embedding_batches_completed=completed, total_embedding_batches=total, vectors_stored=0)
        embeddings = get_embeddings(documents, progress_callback=embedding_progress)

        progress("Storing vectors", files_discovered=len(files), files_processed=len(files), chunks_created=len(chunks), embedding_batches_completed=(len(documents) + settings.embedding_batch_size - 1) // settings.embedding_batch_size, total_embedding_batches=(len(documents) + settings.embedding_batch_size - 1) // settings.embedding_batch_size, vectors_stored=0)
        try:
            store = VectorStore(project_id=project_id)
            store.add_documents(documents, embeddings, metadatas, ids)
            store.remove_documents_except(ids)
            vector_count = store.count()
        except Exception as exc:
            raise ChromaIndexError("The local vector index could not be updated. Check disk space and retry.") from exc
        if vector_count != len(chunks):
            raise ChromaIndexError(f"Expected {len(chunks)} vectors after indexing, but found {vector_count}.")
        progress("Updating project metadata", files_discovered=len(files), files_processed=len(files), chunks_created=len(chunks), embedding_batches_completed=(len(documents) + settings.embedding_batch_size - 1) // settings.embedding_batch_size, total_embedding_batches=(len(documents) + settings.embedding_batch_size - 1) // settings.embedding_batch_size, vectors_stored=vector_count)
    except EmptyProjectError:
        status = "failed" if previous_counts["chunk_count"] else "empty"
        current = (get_project(project_id) or {}).get("indexing_progress", {})
        message = "No supported, non-empty code files were found."
        update_project(project_id, status=status, file_count=previous_counts["file_count"], chunk_count=previous_counts["chunk_count"], indexing_error=message, indexing_progress={**current, "stage": "Failed"})
        logger.warning("Project indexing failed project_id=%s stage=chunking error_category=empty duration_seconds=%.2f", project_id, time.monotonic() - started)
        raise
    except Exception as exc:
        current = (get_project(project_id) or {}).get("indexing_progress", {})
        if isinstance(exc, EmbeddingUnavailableError):
            message = str(exc)
        elif isinstance(exc, EmbeddingError):
            message = str(exc)
        elif isinstance(exc, ChromaIndexError):
            message = str(exc)
        else:
            stage = current.get("stage", "indexing")
            message = f"Indexing failed during {stage.lower()} ({type(exc).__name__}). Check backend logs and retry."
        update_project(project_id, status="failed", file_count=previous_counts["file_count"], chunk_count=previous_counts["chunk_count"], indexing_error=message, indexing_progress={**current, "stage": "Failed"})
        logger.warning("Project indexing failed project_id=%s stage=%s error_category=%s duration_seconds=%.2f", project_id, current.get("stage", "unknown"), type(exc).__name__, time.monotonic() - started)
        raise

    updated = update_project(
        project_id,
        status="indexed",
        file_count=file_count,
        chunk_count=len(chunks),
        indexing_error=None,
        indexing_progress={**(get_project(project_id) or {}).get("indexing_progress", {}), "stage": "Completed", "files_discovered": len(files), "files_processed": len(files), "chunks_created": len(chunks), "vectors_stored": vector_count},
    )
    logger.info("Project indexing completed project_id=%s files=%s chunks=%s vectors=%s duration_seconds=%.2f", project_id, file_count, len(chunks), vector_count, time.monotonic() - started)
    return {
        **updated,
        "files_indexed": file_count,
        "chunks_indexed": len(chunks),
        "vectors_stored": vector_count,
    }

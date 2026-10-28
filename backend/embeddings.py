"""Ollama embedding client with bounded batches and transient retries."""
import math
import time
from numbers import Real

import requests

from backend.settings import settings


OLLAMA_URL = f"{settings.ollama_url}/api/embed"
EMBEDDING_MODEL = settings.embedding_model
EXPECTED_DIMENSIONS = 768 if "nomic-embed-text" in EMBEDDING_MODEL else None
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """An embedding request failed or Ollama returned invalid vectors."""


class EmbeddingUnavailableError(EmbeddingError):
    """Ollama could not be reached or timed out."""


def _embed_batch(texts):
    for attempt in range(settings.embedding_max_retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": EMBEDDING_MODEL, "input": texts},
                timeout=settings.embedding_timeout,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= settings.embedding_max_retries:
                if isinstance(exc, requests.Timeout):
                    message = "Ollama timed out while generating code embeddings after automatic retries. Retry indexing in a moment."
                else:
                    message = "Could not reach Ollama while generating code embeddings. Check that Ollama is running and retry."
                raise EmbeddingUnavailableError(message) from exc
            time.sleep(min(2 ** attempt, 8))
            continue
        except requests.RequestException as exc:
            raise EmbeddingError("The embedding request could not be sent to Ollama.") from exc

        if response.status_code in RETRYABLE_STATUS_CODES:
            if attempt >= settings.embedding_max_retries:
                raise EmbeddingUnavailableError(
                    f"Ollama returned HTTP {response.status_code} after {attempt + 1} attempts. Retry indexing in a moment."
                )
            time.sleep(min(2 ** attempt, 8))
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EmbeddingError(f"Ollama rejected the embedding request (HTTP {response.status_code}).") from exc

        try:
            embeddings = response.json()["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("Ollama returned an invalid embedding response.") from exc
        _validate_embeddings(embeddings, len(texts))
        return embeddings


def _validate_embeddings(embeddings, expected_count):
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        actual = len(embeddings) if isinstance(embeddings, list) else "non-list"
        raise EmbeddingError(f"Ollama returned {actual} embeddings for {expected_count} input texts.")
    dimensions = set()
    for index, vector in enumerate(embeddings):
        if not isinstance(vector, (list, tuple)) or not vector:
            raise EmbeddingError(f"Embedding {index + 1} is empty or malformed.")
        if any(not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) for value in vector):
            raise EmbeddingError(f"Embedding {index + 1} contains a non-numeric or non-finite value.")
        dimensions.add(len(vector))
    if len(dimensions) != 1:
        raise EmbeddingError("Ollama returned embeddings with inconsistent dimensions.")
    dimension = next(iter(dimensions))
    if EXPECTED_DIMENSIONS and dimension != EXPECTED_DIMENSIONS:
        raise EmbeddingError(f"Expected {EXPECTED_DIMENSIONS}-dimensional embeddings from {EMBEDDING_MODEL}, received {dimension}.")


def get_embeddings(texts, progress_callback=None):
    """Return vectors in input order, sending bounded batches to Ollama."""
    if not texts:
        return []
    batch_size = settings.embedding_batch_size
    batches = [texts[start:start + batch_size] for start in range(0, len(texts), batch_size)]
    combined = []
    for batch_number, batch in enumerate(batches, start=1):
        combined.extend(_embed_batch(batch))
        if progress_callback:
            progress_callback(batch_number, len(batches))
    _validate_embeddings(combined, len(texts))
    return combined

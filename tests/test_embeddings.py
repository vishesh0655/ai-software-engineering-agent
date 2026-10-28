from types import SimpleNamespace

import pytest
import requests

from backend import embeddings


def _settings(monkeypatch, *, batch_size=4, retries=0):
    monkeypatch.setattr(
        embeddings,
        "settings",
        SimpleNamespace(
            embedding_batch_size=batch_size,
            embedding_timeout=10,
            embedding_max_retries=retries,
        ),
    )


class _Response:
    def __init__(self, vectors=None, status_code=200):
        self._vectors = vectors if vectors is not None else [[0.0] * 768]
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return {"embeddings": self._vectors}


def test_single_text_returns_one_768_dimension_vector(monkeypatch):
    _settings(monkeypatch)
    calls = []
    monkeypatch.setattr(embeddings.requests, "post", lambda _url, **kwargs: (calls.append(kwargs) or _Response()))

    result = embeddings.get_embeddings(["one source file"])

    assert len(result) == 1
    assert len(result[0]) == 768
    assert calls[0]["json"]["input"] == ["one source file"]
    assert calls[0]["timeout"] == 10


def test_multiple_texts_are_batched_in_order_and_report_progress(monkeypatch):
    _settings(monkeypatch, batch_size=2)
    calls = []

    def post(_url, *, json, timeout):
        calls.append((json["input"], timeout))
        return _Response([[float(ord(text))] * 768 for text in json["input"]])

    monkeypatch.setattr(embeddings.requests, "post", post)
    progress = []
    result = embeddings.get_embeddings(["a", "b", "c"], lambda done, total: progress.append((done, total)))

    assert [vector[0] for vector in result] == [97.0, 98.0, 99.0]
    assert [len(batch) for batch, _ in calls] == [2, 1]
    assert progress == [(1, 2), (2, 2)]


def test_empty_input_skips_ollama(monkeypatch):
    _settings(monkeypatch)
    monkeypatch.setattr(embeddings.requests, "post", lambda *_args, **_kwargs: pytest.fail("unexpected request"))
    assert embeddings.get_embeddings([]) == []


@pytest.mark.parametrize("transient", [requests.Timeout("temporary timeout"), requests.ConnectionError("temporary connection error")])
def test_connection_and_timeout_errors_retry_then_succeed(monkeypatch, transient):
    _settings(monkeypatch, retries=1)
    attempts = []

    def post(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise transient
        return _Response()

    monkeypatch.setattr(embeddings.requests, "post", post)
    monkeypatch.setattr(embeddings.time, "sleep", lambda _seconds: None)
    assert len(embeddings.get_embeddings(["hello"])) == 1
    assert len(attempts) == 2


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_http_status_retries(monkeypatch, status):
    _settings(monkeypatch, retries=1)
    attempts = []

    def post(*_args, **_kwargs):
        attempts.append(True)
        return _Response(status_code=status) if len(attempts) == 1 else _Response()

    monkeypatch.setattr(embeddings.requests, "post", post)
    monkeypatch.setattr(embeddings.time, "sleep", lambda _seconds: None)
    assert len(embeddings.get_embeddings(["hello"])) == 1
    assert len(attempts) == 2


def test_permanent_http_error_is_not_retried(monkeypatch):
    _settings(monkeypatch, retries=3)
    attempts = []
    monkeypatch.setattr(embeddings.requests, "post", lambda *_args, **_kwargs: (attempts.append(True) or _Response(status_code=400)))

    with pytest.raises(embeddings.EmbeddingError, match="HTTP 400"):
        embeddings.get_embeddings(["hello"])
    assert len(attempts) == 1


def test_exhausted_connection_retries_raise_clear_error(monkeypatch):
    _settings(monkeypatch, retries=2)
    attempts = []
    monkeypatch.setattr(embeddings.requests, "post", lambda *_args, **_kwargs: (attempts.append(True) or (_ for _ in ()).throw(requests.ConnectionError("offline"))))
    monkeypatch.setattr(embeddings.time, "sleep", lambda _seconds: None)

    with pytest.raises(embeddings.EmbeddingUnavailableError, match="Could not reach Ollama"):
        embeddings.get_embeddings(["hello"])
    assert len(attempts) == 3


def test_wrong_dimensions_are_rejected(monkeypatch):
    _settings(monkeypatch)
    monkeypatch.setattr(embeddings.requests, "post", lambda *_args, **_kwargs: _Response([[0.0] * 5]))
    with pytest.raises(embeddings.EmbeddingError, match="768-dimensional"):
        embeddings.get_embeddings(["hello"])


def test_mismatched_result_count_is_rejected(monkeypatch):
    _settings(monkeypatch)
    monkeypatch.setattr(embeddings.requests, "post", lambda *_args, **_kwargs: _Response([]))
    with pytest.raises(embeddings.EmbeddingError, match="0 embeddings for 1 input"):
        embeddings.get_embeddings(["hello"])


def test_non_finite_or_non_numeric_values_are_rejected(monkeypatch):
    _settings(monkeypatch)
    monkeypatch.setattr(embeddings.requests, "post", lambda *_args, **_kwargs: _Response([[float("nan")] * 768]))
    with pytest.raises(embeddings.EmbeddingError, match="non-numeric or non-finite"):
        embeddings.get_embeddings(["hello"])

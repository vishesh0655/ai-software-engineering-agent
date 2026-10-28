import pytest

from backend import llm
from backend.agent.prompts import build_prompt


class _Response:
    ok = True
    status_code = 200

    def __init__(self, body):
        self.body = body
        self.text = ""

    def json(self):
        return self.body


def test_empty_ollama_response_raises_safe_generation_error(monkeypatch, caplog):
    monkeypatch.setattr(llm.requests, "post", lambda *_args, **_kwargs: _Response({
        "response": "", "done": True, "done_reason": "stop", "prompt_eval_count": 42,
        "eval_count": 0, "total_duration": 100,
    }))

    with pytest.raises(llm.ModelGenerationError, match="empty model response.*eval_count=0"):
        llm.generate("prompt", response_format="json")

    assert "response=" not in caplog.text
    assert "eval_count" in caplog.text


def test_ollama_non_text_response_raises_generation_error(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *_args, **_kwargs: _Response({"response": None}))

    with pytest.raises(llm.ModelGenerationError, match="no text field"):
        llm.generate("prompt", response_format="json")


def test_ollama_retries_empty_response_once(monkeypatch, caplog):
    responses = iter([
        _Response({"response": "  ", "done": True, "done_reason": "stop", "eval_count": 0}),
        _Response({"response": "MODEL_OK", "done": True, "done_reason": "stop", "eval_count": 1}),
    ])

    monkeypatch.setattr(llm.requests, "post", lambda *_args, **_kwargs: next(responses))

    assert llm.generate("retry prompt", response_format="json") == "MODEL_OK"
    assert "empty model response" in caplog.text.lower()


def test_agent_prompt_distinguishes_observations_from_action_schema():
    prompt = build_prompt("sample", "review my code", [{
        "tool": "search_code",
        "arguments": {"query": "spam"},
        "success": True,
        "result": "[{'file': 'app.py'}]",
    }])

    assert "READ ONLY" in prompt
    assert "NEVER emit an observation-shaped object" in prompt
    assert '"action":"search_code"' in prompt
    assert '"action":"final","answer"' in prompt

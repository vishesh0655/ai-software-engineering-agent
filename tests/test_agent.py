import json

import pytest
from fastapi.testclient import TestClient

from backend.agent import agent as agent_module
from backend.agent.agent import AgentError, AgentLimits, run_agent
from backend.api import app
from backend.tools import success_result


def _responses(monkeypatch, *responses):
    pending = iter(responses)
    prompts = []

    def fake_generate(prompt, **_kwargs):
        prompts.append(prompt)
        return next(pending)

    monkeypatch.setattr(agent_module, "generate", fake_generate)
    return prompts


def _fake_tools(monkeypatch):
    invoked = []

    def fake_search_code(**arguments):
        invoked.append(("search_code", arguments))
        return success_result("search_code", {"results": [{"file": "auth.py"}]})

    def fake_find_symbol(**arguments):
        invoked.append(("find_symbol", arguments))
        return success_result("find_symbol", {"matches": [{"file": "auth.py", "symbol": arguments["symbol"]}]})

    def fake_read_file(**arguments):
        invoked.append(("read_file", arguments))
        return success_result("read_file", {"content": "verified source"})

    tools = {
        "search_code": fake_search_code,
        "find_symbol": fake_find_symbol,
        "read_file": fake_read_file,
    }
    monkeypatch.setattr(agent_module, "get_tool", tools.get)
    return invoked


def test_agent_accepts_successful_final_response_after_gathering_evidence(monkeypatch):
    _fake_tools(monkeypatch)
    iterations = []
    _responses(
        monkeypatch,
        '{"action":"search_code","arguments":{"query":"authentication"}}',
        json.dumps({"action": "final", "answer": "Evidence is insufficient."}),
    )
    result = run_agent("tradex", "Where is auth?", on_iteration=iterations.append)
    assert result["answer"] == "Evidence is insufficient."
    assert len(result["tool_calls"]) == 1
    assert result["iterations"] == 2
    assert iterations == [1, 2]


def test_agent_calls_tool_then_returns_grounded_final(monkeypatch):
    invoked = _fake_tools(monkeypatch)
    prompts = _responses(
        monkeypatch,
        '{"action":"search_code","arguments":{"query":"authentication"}}',
        '{"action":"final","answer":"Authentication code is in auth.py."}',
    )
    result = run_agent("tradex", "Where is authentication implemented?")
    assert result["answer"].endswith("auth.py.")
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "search_code"
    assert invoked[0][1]["project_id"] == "tradex"
    assert "search_code" in prompts[1]


def test_agent_can_make_multiple_tool_calls(monkeypatch):
    invoked = _fake_tools(monkeypatch)
    _responses(
        monkeypatch,
        '{"action":"search_code","arguments":{"query":"JWT tokens"}}',
        '{"action":"find_symbol","arguments":{"symbol":"create_access_token"}}',
        '{"action":"read_file","arguments":{"file_path":"auth.py"}}',
        '{"action":"final","answer":"Tokens are generated in auth.py."}',
    )
    result = run_agent("tradex", "Where are JWT tokens generated?")
    assert [call["tool"] for call in result["tool_calls"]] == [
        "search_code", "find_symbol", "read_file",
    ]
    assert len(invoked) == 3
    assert result["iterations"] == 4


def test_agent_records_tool_failure_and_continues_from_the_observation(monkeypatch):
    def broken_search(**_arguments):
        raise PermissionError("private filesystem detail")

    monkeypatch.setattr(agent_module, "get_tool", lambda name: broken_search if name == "search_code" else None)
    _responses(
        monkeypatch,
        '{"action":"search_code","arguments":{"query":"authentication"}}',
        '{"action":"final","answer":"The search tool failed; I could not verify the implementation."}',
    )

    result = run_agent("sample", "Where is authentication implemented?")

    assert result["tool_calls"][0]["success"] is False
    assert "PermissionError" in result["tool_calls"][0]["result"]
    assert "private filesystem detail" not in result["tool_calls"][0]["result"]
    assert "could not verify" in result["answer"]


@pytest.mark.parametrize("bad_response", [
    "not JSON",
    '{"action":"run_shell","arguments":{"command":"whoami"}}',
    '{"action":"search_code","arguments":{"query":"auth","command":"whoami"}}',
    '{"tool":"search_code","success":true,"result":"evidence"}',
])
def test_agent_stops_after_repeated_invalid_actions(monkeypatch, bad_response):
    _responses(monkeypatch, bad_response, bad_response)
    with pytest.raises(AgentError, match="repeated invalid actions"):
        run_agent("tradex", "Inspect auth")


def test_empty_model_response_has_safe_parse_diagnostics():
    from backend.agent.protocol import ProtocolError, validate_action

    with pytest.raises(ProtocolError, match="empty or whitespace") as error:
        validate_action("  ", "tradex")

    assert error.value.response_length == 2
    assert error.value.response_start_kind == "empty"


def test_prompt_separates_action_envelope_from_tool_observation():
    from backend.agent.prompts import build_prompt

    prompt = build_prompt("sample", "review my code", [{
        "observation": '{"tool":"search_code","success":true,"result":"evidence"}',
    }])

    assert '"action":"<registered tool name>","arguments"' in prompt
    assert '"action":"final","answer"' in prompt
    assert "never copy them into your action response" in prompt


def test_agent_synthesizes_final_answer_when_iteration_limit_is_reached_with_evidence(monkeypatch):
    _fake_tools(monkeypatch)
    _responses(
        monkeypatch,
        '{"action":"search_code","arguments":{"query":"auth"}}',
        '{"action":"final","answer":"done"}',
    )
    result = run_agent("tradex", "Inspect auth", limits=AgentLimits(max_iterations=1))
    assert result["answer"] == "done"
    assert len(result["tool_calls"]) == 1
    assert result["iterations"] == 1


def test_agent_rejects_observation_like_output_even_when_it_looks_like_tool_result(monkeypatch):
    _fake_tools(monkeypatch)
    _responses(
        monkeypatch,
        '{"tool":"search_code","success":true,"result":"evidence"}',
        '{"tool":"search_code","success":true,"result":"evidence"}',
    )
    with pytest.raises(AgentError, match="repeated invalid actions"):
        run_agent("tradex", "Inspect auth")


def test_agent_enforces_tool_call_limit_but_allows_final_after_limit(monkeypatch):
    _fake_tools(monkeypatch)
    _responses(
        monkeypatch,
        '{"action":"search_code","arguments":{"query":"auth"}}',
        '{"action":"final","answer":"done"}',
    )
    result = run_agent("tradex", "Inspect auth", limits=AgentLimits(max_tool_calls=1))
    assert len(result["tool_calls"]) == 1


def test_agent_tool_call_limit_stops_another_tool_action(monkeypatch):
    _fake_tools(monkeypatch)
    _responses(
        monkeypatch,
        '{"action":"search_code","arguments":{"query":"auth"}}',
        '{"action":"search_code","arguments":{"query":"tokens"}}',
    )
    with pytest.raises(AgentError, match="maximum number of tool calls"):
        run_agent("tradex", "Inspect auth", limits=AgentLimits(max_tool_calls=1))


def test_agent_api_validates_project_and_preserves_existing_routes(monkeypatch, tmp_path):
    from backend import api

    project = {"id": "tradex", "name": "TradeX", "path": str(tmp_path), "status": "indexed"}
    monkeypatch.setattr(api, "get_project", lambda project_id: project if project_id == "tradex" else None)
    monkeypatch.setattr(api, "list_projects", lambda: [project])
    monkeypatch.setattr(api, "ask_codebase", lambda question, project_id=None: {
        "answer": "legacy answer", "sources": [],
    })
    monkeypatch.setattr(api, "run_agent", lambda project_id, question: {
        "project_id": project_id, "answer": "agent answer", "tool_calls": [], "iterations": 1,
    })
    monkeypatch.setattr(api, "register_project", lambda path: {**project, "path": str(path)})
    monkeypatch.setattr(api, "index_registered_project", lambda project_id: {
        "id": project_id, "status": "indexed", "file_count": 1, "chunk_count": 1,
    })

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/tools").status_code == 200
    assert client.get("/projects").json()["projects"][0]["id"] == "tradex"
    assert client.get("/projects/tradex").json()["id"] == "tradex"
    indexed = client.post("/projects/index", json={"project_path": str(tmp_path)})
    assert indexed.status_code == 200
    assert indexed.json()["chunk_count"] == 1
    assert client.post("/ask", json={"question": "legacy"}).json()["answer"] == "legacy answer"
    response = client.post("/agent", json={"project_id": "tradex", "question": "inspect"})
    assert response.status_code == 200
    assert response.json()["answer"] == "agent answer"
    assert client.post("/agent", json={"project_id": "missing", "question": "inspect"}).status_code == 404
    assert client.post("/agent", json={"project_id": "tradex", "question": " "}).status_code == 400


def test_agent_protocol_is_limited_to_registered_tools():
    from backend.agent.protocol import ProtocolError, validate_action

    with pytest.raises(ProtocolError, match="Unknown action"):
        validate_action('{"action":"run_shell","arguments":{"command":"x"}}', "tradex")
    with pytest.raises(ProtocolError, match="must not select"):
        validate_action('{"action":"search_code","arguments":{"query":"auth","project_id":"other"}}', "tradex")
    with pytest.raises(ProtocolError, match="Malformed tool action: action is required"):
        validate_action('{"arguments":{}}', "tradex")
    action, args = validate_action('{"action":"inspect_project"}', "tradex")
    assert action == "inspect_project"
    assert args == {"project_id": "tradex"}
    with pytest.raises(ProtocolError, match="Tool arguments failed validation: query is required"):
        validate_action('{"action":"search_code"}', "tradex")

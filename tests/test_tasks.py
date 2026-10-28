import json
import logging
import pytest
from fastapi.testclient import TestClient

from backend import project_manager
from backend.agent import tasks
from backend.agent.agent import AgentError
from backend.agent.protocol import ProtocolError
from backend.api import app
from backend.project_manager import register_project
from backend.tools import success_result
from backend.vector_store import VectorStore


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "sample-project"
    root.mkdir()
    source = root / "app.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", tmp_path / "projects.json")
    monkeypatch.setattr(tasks, "TASKS_FILE", tmp_path / "agent-tasks.json")
    monkeypatch.setattr(tasks, "BACKUP_ROOT", tmp_path / "backups")
    registered = register_project(root)
    return root, source, registered["id"]


def setup_agent(monkeypatch, new_content="def value():\n    return 2\n"):
    calls = []

    def fake_run_agent(project_id, request, *, on_tool_call=None, on_iteration=None):
        calls.append((project_id, request))
        if on_iteration:
            on_iteration(2)
        if on_tool_call:
            on_tool_call({
                "iteration": 1, "tool": "read_file", "success": True,
                "elapsed_ms": 1, "result": "safe test evidence",
            })
        return {
            "project_id": project_id,
            "answer": "The current implementation returns the wrong value.",
            "tool_calls": [{"tool": "read_file", "success": True, "result": "app.py source"}],
            "iterations": 2,
        }

    def fake_generate(_prompt, response_format=None):
        assert response_format == "json"
        return json.dumps({"changes": [{
            "path": "app.py", "old_text": "return 1",
            "new_text": "return " + new_content.split("return ", 1)[1].splitlines()[0],
            "reason": "Correct the returned value.",
        }]})

    monkeypatch.setattr(tasks, "run_agent", fake_run_agent)
    monkeypatch.setattr(tasks, "generate", fake_generate)
    return calls


def test_investigation_and_review_create_persisted_tasks_without_proposals(project, monkeypatch):
    _, _, project_id = project
    calls = setup_agent(monkeypatch)
    task = tasks.create_task(project_id, "Review this code", "auto")
    assert task["mode"] == "review"
    assert task["status"] == "investigated"
    assert task["proposed_changes"] == []
    assert "code review" in calls[0][1].lower()
    loaded = tasks.get_task(task["task_id"])
    assert loaded["request"] == "Review this code"
    assert loaded["tool_calls"][0]["tool"] == "read_file"
    assert tasks._infer_mode("Review the code and fix the bug", "auto") == "change"
    review_prompt = tasks._investigation_request("review my code", "review")
    assert "This is review-only" in review_prompt
    assert "do not generate code, patches, or diffs" in review_prompt


def test_debugging_task_proposes_server_generated_unified_diff(project, monkeypatch):
    _, source, project_id = project
    setup_agent(monkeypatch)
    task = tasks.create_task(project_id, "Find and fix this bug")
    assert task["mode"] == "change"
    assert task["status"] == "awaiting_approval"
    proposed = task["proposed_changes"][0]
    assert proposed["path"] == "app.py"
    assert proposed["approval_status"] == "pending"
    assert "--- a/app.py" in proposed["diff"]
    assert "+++ b/app.py" in proposed["diff"]
    assert "-    return 1" in proposed["diff"]
    assert "    return 2" in proposed["diff"]
    assert source.read_text(encoding="utf-8").endswith("return 1\n")


def test_approval_applies_change_runs_registered_tests_and_rollback_restores_backup(project, monkeypatch):
    _, source, project_id = project
    setup_agent(monkeypatch)
    test_calls = []

    def fake_test_tool(**kwargs):
        test_calls.append(kwargs)
        return success_result("run_tests", {
            "passed": True, "exit_code": 0, "output": "2 passed", "duration_seconds": 0.1,
        })

    monkeypatch.setattr(tasks, "get_tool", lambda name: fake_test_tool if name == "run_tests" else None)
    task = tasks.create_task(project_id, "Fix the function")
    assert source.read_text(encoding="utf-8").endswith("return 1\n")
    approved = tasks.approve_task(task["task_id"])
    assert approved["status"] == "completed"
    assert approved["test_results"][0]["passed"] is True
    assert test_calls == [{"project_id": project_id}]
    assert source.read_text(encoding="utf-8").endswith("return 2\n")

    monkeypatch.setattr(VectorStore, "delete_project_collection", lambda _project_id: None)
    project_manager.delete_project(project_id)
    rolled_back = tasks.rollback_task(task["task_id"])
    assert rolled_back["status"] == "rolled_back"
    assert source.read_text(encoding="utf-8").endswith("return 1\n")
    assert "project_path" not in rolled_back


def test_rejection_never_modifies_source(project, monkeypatch):
    _, source, project_id = project
    setup_agent(monkeypatch)
    task = tasks.create_task(project_id, "Fix the function")
    rejected = tasks.reject_task(task["task_id"])
    assert rejected["status"] == "rejected"
    assert rejected["approval_status"] == "rejected"
    assert source.read_text(encoding="utf-8").endswith("return 1\n")


@pytest.mark.parametrize("bad_path", ["../outside.py", "C:\\Windows\\win.ini", "/tmp/outside.py"])
def test_proposals_reject_paths_outside_project(project, monkeypatch, bad_path):
    _, _, project_id = project
    setup_agent(monkeypatch)

    def malicious(_prompt, response_format=None):
        return json.dumps({"changes": [{
            "path": bad_path, "old_text": "return 1", "new_text": "return 2", "reason": "bad",
        }]})

    monkeypatch.setattr(tasks, "generate", malicious)
    with pytest.raises(tasks.TaskError, match="relative|inside"):
        tasks.create_task(project_id, "Fix the function")


def test_test_failure_creates_followup_proposal_that_requires_approval(project, monkeypatch):
    _, source, project_id = project
    calls = setup_agent(monkeypatch, "def value():\n    return 2\n")
    run_count = 0

    def fake_test_tool(**_kwargs):
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            return success_result("run_tests", {
                "passed": False, "exit_code": 1, "output": "AssertionError", "duration_seconds": 0.1,
            })
        return success_result("run_tests", {
            "passed": True, "exit_code": 0, "output": "2 passed", "duration_seconds": 0.1,
        })

    monkeypatch.setattr(tasks, "get_tool", lambda name: fake_test_tool if name == "run_tests" else None)
    task = tasks.create_task(project_id, "Find and fix the failing function")
    monkeypatch.setattr(tasks, "generate", lambda *_args, **_kwargs: json.dumps({"changes": [{
        "path": "app.py", "old_text": "return 2", "new_text": "return 3",
        "reason": "Correct the test failure.",
    }]}))
    failed = tasks.approve_task(task["task_id"])
    assert failed["test_results"][0]["passed"] is False
    assert failed["status"] == "awaiting_approval", failed.get("result")
    assert failed["approval_status"] == "pending"
    assert "test failure" in calls[-1][1].lower()
    assert source.read_text(encoding="utf-8").endswith("return 2\n")

    completed = tasks.approve_task(task["task_id"])
    assert completed["status"] == "completed"
    assert [result["passed"] for result in completed["test_results"]] == [False, True]
    assert source.read_text(encoding="utf-8").endswith("return 3\n")
    assert tasks.rollback_task(task["task_id"])["status"] == "rolled_back"
    assert source.read_text(encoding="utf-8").endswith("return 1\n")


def test_failed_tests_never_report_success(project, monkeypatch):
    _, _, project_id = project
    setup_agent(monkeypatch)

    def failed_test(**_kwargs):
        return success_result("run_tests", {
            "passed": False, "exit_code": 1, "output": "failure", "duration_seconds": 0.1,
        })

    # End the repair loop after the initial failed run by refusing a follow-up proposal.
    monkeypatch.setattr(tasks, "get_tool", lambda name: failed_test if name == "run_tests" else None)
    task = tasks.create_task(project_id, "Fix the function")
    monkeypatch.setattr(tasks, "generate", lambda *_args, **_kwargs: '{"changes":[]}')
    result = tasks.approve_task(task["task_id"])
    assert result["status"] == "tests_failed"
    assert result["test_results"][0]["passed"] is False
    assert "passed" not in result["result"].lower()


def test_test_failure_with_unavailable_followup_agent_is_reported_as_failure(project, monkeypatch):
    _, _, project_id = project
    setup_agent(monkeypatch)

    def failed_test(**_kwargs):
        return success_result("run_tests", {
            "passed": False, "exit_code": 1, "output": "AssertionError", "duration_seconds": 0.1,
        })

    monkeypatch.setattr(tasks, "get_tool", lambda name: failed_test if name == "run_tests" else None)
    task = tasks.create_task(project_id, "Fix the function")
    monkeypatch.setattr(tasks, "run_agent", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model offline")))
    result = tasks.approve_task(task["task_id"])
    assert result["status"] == "tests_failed"
    assert result["test_results"][0]["passed"] is False
    assert "model offline" in result["result"]


def test_fix_test_retry_loop_stops_at_configured_attempt_limit(project, monkeypatch):
    _, source, project_id = project
    setup_agent(monkeypatch)
    def failing_tests(**_kwargs):
        return success_result("run_tests", {
            "passed": False, "exit_code": 1, "output": "still failing", "duration_seconds": 0.1,
        })

    monkeypatch.setattr(tasks, "get_tool", lambda name: failing_tests if name == "run_tests" else None)
    task = tasks.create_task(project_id, "Fix the function")
    generated = iter([
        json.dumps({"changes": [{"path": "app.py", "old_text": "return 2", "new_text": "return 3", "reason": "retry 2"}]}),
        json.dumps({"changes": [{"path": "app.py", "old_text": "return 3", "new_text": "return 4", "reason": "retry 3"}]}),
    ])
    monkeypatch.setattr(tasks, "generate", lambda *_args, **_kwargs: next(generated))
    for _ in range(2):
        task = tasks.approve_task(task["task_id"])
        assert task["status"] == "awaiting_approval"
    task = tasks.approve_task(task["task_id"])
    assert task["status"] == "tests_failed"
    assert task["attempts"] == tasks.MAX_FIX_ATTEMPTS
    assert len(task["test_results"]) == tasks.MAX_FIX_ATTEMPTS
    assert "maximum approved fix attempts" in task["result"]
    assert source.read_text(encoding="utf-8").endswith("return 4\n")


def test_approval_detects_stale_file_and_refuses_to_overwrite(project, monkeypatch):
    _, source, project_id = project
    setup_agent(monkeypatch)
    task = tasks.create_task(project_id, "Fix the function")
    source.write_text("manual change\n", encoding="utf-8")
    result = tasks.approve_task(task["task_id"])
    assert result["status"] == "apply_failed"
    assert "changed since proposal" in result["result"].lower()
    assert source.read_text(encoding="utf-8") == "manual change\n"


def test_partial_apply_failure_restores_every_file(project, monkeypatch):
    root, source, project_id = project
    second = root / "other.py"
    second.write_text("def other():\n    return 10\n", encoding="utf-8")
    setup_agent(monkeypatch)

    def proposal(_prompt, response_format=None):
        return json.dumps({"changes": [
            {"path": "app.py", "old_text": "return 1", "new_text": "return 2", "reason": "fix"},
            {"path": "other.py", "old_text": "return 10", "new_text": "return 20", "reason": "fix"},
        ]})

    monkeypatch.setattr(tasks, "generate", proposal)
    task = tasks.create_task(project_id, "Fix both functions")
    real_write = tasks._write_atomic

    def fail_second(target, content):
        if target.name == "other.py":
            raise OSError("simulated disk failure")
        return real_write(target, content)

    monkeypatch.setattr(tasks, "_write_atomic", fail_second)
    result = tasks.approve_task(task["task_id"])
    assert result["status"] == "apply_failed"
    assert "restored" in result["result"]
    assert source.read_text(encoding="utf-8").endswith("return 1\n")
    assert second.read_text(encoding="utf-8").endswith("return 10\n")


def test_complete_workflow_runs_real_pytest_on_temporary_project(project, monkeypatch):
    root, source, project_id = project
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "from app import value\n\ndef test_fixed_value():\n    assert value() == 2\n",
        encoding="utf-8",
    )
    setup_agent(monkeypatch)
    task = tasks.create_task(project_id, "Find and fix the incorrect return value")
    assert task["status"] == "awaiting_approval"
    assert task["proposed_changes"][0]["diff"]
    approved = tasks.approve_task(task["task_id"])
    assert approved["status"] == "completed"
    assert approved["test_results"][0]["passed"] is True
    assert "1 passed" in approved["test_results"][0]["output"]
    assert source.read_text(encoding="utf-8").endswith("return 2\n")
    rolled_back = tasks.rollback_task(task["task_id"])
    assert rolled_back["status"] == "rolled_back"
    assert source.read_text(encoding="utf-8").endswith("return 1\n")


def test_task_api_create_status_approve_reject_and_rollback(project, monkeypatch):
    _, source, project_id = project
    setup_agent(monkeypatch)
    monkeypatch.setattr(tasks, "get_tool", lambda name: (
        lambda **_kwargs: success_result("run_tests", {
            "passed": True, "exit_code": 0, "output": "1 passed", "duration_seconds": 0.1,
        })
    ) if name == "run_tests" else None)
    client = TestClient(app)
    created = client.post("/tasks", json={"project_id": project_id, "request": "Fix the function"})
    assert created.status_code == 201
    task_id = created.json()["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["status"] == "awaiting_approval"
    approved = client.post(f"/tasks/{task_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "completed"
    assert client.post(f"/tasks/{task_id}/rollback").json()["status"] == "rolled_back"
    assert source.read_text(encoding="utf-8").endswith("return 1\n")

    second = client.post("/tasks", json={"project_id": project_id, "request": "Fix this", "mode": "change"})
    second_id = second.json()["task_id"]
    rejected = client.post(f"/tasks/{second_id}/reject")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert client.get("/tasks/not-a-task").status_code == 404


def test_background_task_api_returns_task_and_persists_progress_events(project, monkeypatch):
    _, _, project_id = project

    def fake_run_agent(_project_id, _request, *, on_tool_call=None, **_kwargs):
        if on_tool_call:
            on_tool_call({"tool": "inspect_project", "success": True, "elapsed_ms": 1})
        return {
            "answer": "Inspected the temporary project.",
            "tool_calls": [{"tool": "inspect_project", "success": True, "elapsed_ms": 1}],
            "iterations": 2,
        }

    monkeypatch.setattr(tasks, "run_agent", fake_run_agent)
    monkeypatch.setattr(tasks, "generate", lambda *_args, **_kwargs: json.dumps({"changes": [{
        "path": "app.py", "old_text": "return 1", "new_text": "return 2", "reason": "fix",
    }]}))
    response = TestClient(app).post(
        "/tasks?background=true",
        json={"project_id": project_id, "request": "Fix this bug", "mode": "change"},
    )
    assert response.status_code == 202
    task = tasks.get_task(response.json()["task_id"])
    assert task["status"] == "awaiting_approval"
    assert task["tool_calls"][0]["tool"] == "inspect_project"
    assert any(action["action"] == "tool_call" for action in task["actions"])


def test_background_review_task_finishes_without_proposal(project, monkeypatch):
    _, source, project_id = project
    calls = []

    def fake_run_agent(_project_id, request, *, on_tool_call=None, on_iteration=None):
        calls.append(request)
        if on_iteration:
            on_iteration(2)
        if on_tool_call:
            on_tool_call({"tool": "inspect_project", "success": True, "iteration": 1, "elapsed_ms": 1})
        return {"answer": "Review found no confirmed issue.", "tool_calls": [{"tool": "inspect_project", "success": True}], "iterations": 2}

    monkeypatch.setattr(tasks, "run_agent", fake_run_agent)
    monkeypatch.setattr(tasks, "_make_proposal", lambda *_args, **_kwargs: pytest.fail("Review mode must not generate a proposal"))
    task = tasks.start_task(project_id, "review my code")

    result = tasks.process_task(task["task_id"])

    assert result["mode"] == "review"
    assert result["status"] == "investigated"
    assert result["answer"] == "Review found no confirmed issue."
    assert result["iterations"] == 2
    assert result["proposed_changes"] == []
    assert "code review" in calls[0].lower()
    assert source.read_text(encoding="utf-8").endswith("return 1\n")


def test_background_agent_error_saves_safe_diagnostic_and_logs_traceback(project, monkeypatch, caplog):
    _, source, project_id = project

    def fail_agent(_project_id, _request, *, on_iteration=None, **_kwargs):
        if on_iteration:
            on_iteration(3)
        try:
            json.loads("not valid JSON")
        except json.JSONDecodeError as json_error:
            cause = ProtocolError("The model response was not valid JSON: Expecting value at line 1, column 1")
            cause.__cause__ = json_error
            raise AgentError(f"Agent stopped after repeated invalid actions: {cause}") from cause

    monkeypatch.setattr(tasks, "run_agent", fail_agent)
    task = tasks.start_task(project_id, "review my code")
    caplog.set_level(logging.ERROR, logger="backend.agent.tasks")

    result = tasks.process_task(task["task_id"])

    assert result["status"] == "failed"
    assert result["iterations"] == 3
    assert result["error"]["code"] == "AGENT_INVALID_JSON"
    assert result["error"]["cause_type"] == "ProtocolError"
    assert result["error"]["root_cause_type"] == "JSONDecodeError"
    assert result["error"]["cause_message"].startswith("The model response was not valid JSON:")
    assert "invalid JSON" in result["error"]["message"]
    assert "Traceback (most recent call last)" in caplog.text
    assert task["task_id"] in caplog.text
    assert project_id in caplog.text
    assert "review" in caplog.text
    assert source.read_text(encoding="utf-8").endswith("return 1\n")


def test_background_proposal_failure_is_distinct_from_investigation_failure(project, monkeypatch, caplog):
    _, source, project_id = project
    setup_agent(monkeypatch)
    monkeypatch.setattr(tasks, "_make_proposal", lambda *_args, **_kwargs: (_ for _ in ()).throw(tasks.TaskError("invalid generated proposal")))
    task = tasks.start_task(project_id, "Fix the function", "change")

    result = tasks.process_task(task["task_id"])

    assert result["status"] == "failed"
    assert result["error"]["stage"] == "proposal"
    assert result["error"]["code"] == "PROPOSAL_GENERATION_FAILED"
    assert "structured change proposal" in result["result"]
    assert source.read_text(encoding="utf-8").endswith("return 1\n")

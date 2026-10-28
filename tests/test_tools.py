import subprocess
from types import SimpleNamespace

import pytest

from backend.project_manager import register_project
from backend.tools import registry
from backend.tools import run_tests as run_tests_module
from backend.tools import search_code as search_code_module
from backend.tools.find_symbol import find_symbol
from backend.tools.inspect_project import inspect_project
from backend.tools.read_file import read_file
from backend.tools.run_tests import run_tests
from backend.tools.search_code import search_code


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "sample-repo"
    root.mkdir()
    monkeypatch.setattr("backend.project_manager.PROJECTS_FILE", tmp_path / "projects.json")
    metadata = register_project(root)
    return root, metadata["id"]


def test_search_code_is_project_scoped_and_rejects_empty_query(project, monkeypatch):
    root, project_id = project
    source = root / "src" / "auth.py"
    source.parent.mkdir()
    source.write_text("def login():\n    return True\n", encoding="utf-8")
    observed = {}

    def fake_retrieve(query, **kwargs):
        observed.update(kwargs)
        return [{
            "metadata": {
                "file": str(source), "symbol": "login", "type": "function",
                "start_line": 1, "end_line": 2, "language": "python", "parent": "",
            },
            "content": "def login():\n    return True",
        }]

    monkeypatch.setattr(search_code_module, "retrieve", fake_retrieve)
    result = search_code(project_id, "authentication", 1)
    assert result["success"] is True
    assert result["data"]["results"][0]["file"] == "src/auth.py"
    assert observed["project_id"] == project_id

    empty = search_code(project_id, "   ")
    assert empty["success"] is False
    assert "empty" in empty["error"].lower()


def test_read_file_ranges_secrets_and_missing_paths(project):
    root, project_id = project
    source = root / "module.py"
    source.write_text(
        'API_KEY = "dont-show-this"\nsecond = 2\nthird = 3\n', encoding="utf-8"
    )
    result = read_file(project_id, "module.py", 2, 3)
    assert result["success"] is True
    assert result["data"]["start_line"] == 2
    assert result["data"]["end_line"] == 3
    assert "dont-show-this" not in result["data"]["content"]

    missing = read_file(project_id, "missing.py")
    assert missing["success"] is False
    assert "not found" in missing["error"].lower()

    invalid_range = read_file(project_id, "module.py", 3, 2)
    assert invalid_range["success"] is False


def test_read_file_rejects_traversal_and_absolute_outside_path(project, tmp_path):
    root, project_id = project
    outside = tmp_path / "outside.txt"
    outside.write_text("do not read", encoding="utf-8")

    traversal = read_file(project_id, "../outside.txt")
    absolute = read_file(project_id, str(outside))
    assert traversal["success"] is False
    assert absolute["success"] is False
    assert "inside" in traversal["error"].lower()


def test_tools_reject_unknown_projects_and_non_files(project):
    root, project_id = project
    assert read_file("missing-project", "file.py")["success"] is False
    directory_result = read_file(project_id, ".")
    assert directory_result["success"] is False
    assert "not a file" in directory_result["error"].lower()


def test_find_symbol_uses_ast_and_scopes_to_file(project):
    root, project_id = project
    source = root / "app.py"
    source.write_text(
        "class Auth:\n    async def verify_password(self):\n        return True\n",
        encoding="utf-8",
    )
    matches = find_symbol(project_id, "verify_password")
    assert matches["success"] is True
    assert matches["data"]["matches"] == [{
        "file": "app.py", "symbol": "verify_password", "type": "method",
        "parent": "Auth", "start_line": 2, "end_line": 3,
    }]
    absent = find_symbol(project_id, "not_here", "app.py")
    assert absent["data"]["matches"] == []


def test_find_symbol_supports_basic_text_declarations(project):
    root, project_id = project
    (root / "client.js").write_text(
        "export async function authenticate() { return true; }\n",
        encoding="utf-8",
    )
    result = find_symbol(project_id, "authenticate")
    assert result["success"] is True
    assert result["data"]["matches"][0]["file"] == "client.js"
    assert result["data"]["matches"][0]["type"] == "text_symbol"


def test_inspect_project_reports_current_files(project):
    root, project_id = project
    (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "web.js").write_text("function start() {}\n", encoding="utf-8")
    result = inspect_project(project_id)
    assert result["success"] is True
    assert result["data"]["languages"] == {"javascript": 1, "python": 1}
    assert result["data"]["name"] == "sample-repo"


def test_run_tests_uses_fixed_pytest_command_and_captures_failure(project, monkeypatch):
    root, project_id = project
    (root / "tests").mkdir()
    (root / "tests" / "test_sample.py").write_text("def test_it(): pass\n", encoding="utf-8")
    monkeypatch.setattr(run_tests_module.importlib.util, "find_spec", lambda _name: object())
    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls.update(kwargs)
        return SimpleNamespace(returncode=1, stdout="1 failed", stderr="")

    monkeypatch.setattr(run_tests_module.subprocess, "run", fake_run)
    result = run_tests(project_id)
    assert result["success"] is True
    assert result["data"]["passed"] is False
    assert result["data"]["passed_count"] == 0
    assert result["data"]["failed_count"] == 1
    assert result["data"]["exit_code"] == 1
    assert calls["args"][-3:] == ["-q", "-p", "no:cacheprovider"]
    assert "shell" not in calls
    assert calls["cwd"] == str(root)


def test_run_tests_handles_timeout(project, monkeypatch):
    root, project_id = project
    (root / "test_timeout.py").write_text("def test_it(): pass\n", encoding="utf-8")
    monkeypatch.setattr(run_tests_module.importlib.util, "find_spec", lambda _name: object())

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("pytest", 120, output=b"started")

    monkeypatch.setattr(run_tests_module.subprocess, "run", timeout)
    result = run_tests(project_id)
    assert result["success"] is True
    assert result["data"]["exit_code"] == 124
    assert "exceeded" in result["data"]["output"]


def test_registry_lists_registered_tools_and_handles_unknown():
    names = {tool["name"] for tool in registry.list_tools()}
    assert names == {"search_code", "read_file", "find_symbol", "inspect_project", "run_tests"}
    assert registry.get_tool("missing") is None
    assert registry.get_tool("search_code") is search_code

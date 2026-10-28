import subprocess

from fastapi.testclient import TestClient

from backend import project_manager
from backend.api import app
from backend.git_support import inspect_git
from backend.project_manager import register_project


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, shell=False,
    )


def test_git_inspection_reports_branch_changes_and_diff(tmp_path, monkeypatch):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "app.py").write_text("value = 1\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", "app.py")
    _git(root, "commit", "--quiet", "-m", "initial")
    (root / "app.py").write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", tmp_path / "projects.json")
    project = register_project(root)

    result = inspect_git(project["id"])
    assert result["is_git_repository"] is True
    assert result["branch"]
    assert result["changed_files"] == [{"status": "M", "path": "app.py"}]
    assert "-value = 1" in result["diff"]
    assert "+value = 2" in result["diff"]


def test_git_api_reports_non_repository_without_running_mutations(tmp_path, monkeypatch):
    root = tmp_path / "plain"
    root.mkdir()
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", tmp_path / "projects.json")
    project = register_project(root)
    response = TestClient(app).get(f"/projects/{project['id']}/git")
    assert response.status_code == 200
    assert response.json()["is_git_repository"] is False
    assert response.json()["changed_files"] == []

import json
import threading

from backend import project_manager


def test_project_metadata_write_retries_transient_permission_error(tmp_path, monkeypatch):
    projects_file = tmp_path / "projects.json"
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", projects_file)
    original_replace = project_manager.Path.replace
    attempts = 0

    def replace_with_transient_denial(path, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("file is temporarily in use")
        return original_replace(path, target)

    monkeypatch.setattr(project_manager.Path, "replace", replace_with_transient_denial)

    project_manager._write_projects({"sample": {"id": "sample"}})

    assert attempts == 2
    assert json.loads(projects_file.read_text(encoding="utf-8")) == {
        "sample": {"id": "sample"}
    }
    assert list(tmp_path.glob("projects-*.tmp")) == []


def test_concurrent_project_updates_preserve_both_metadata_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_FILE", tmp_path / "projects.json")
    project_root = tmp_path / "source"
    project_root.mkdir()
    project = project_manager.register_project(project_root)
    first_read_started = threading.Event()
    allow_first_read = threading.Event()
    original_read = project_manager._read_projects

    def delayed_read():
        projects = original_read()
        if threading.current_thread().name == "first-update":
            first_read_started.set()
            assert allow_first_read.wait(timeout=2)
        return projects

    monkeypatch.setattr(project_manager, "_read_projects", delayed_read)
    first = threading.Thread(
        name="first-update",
        target=lambda: project_manager.update_project(project["id"], file_count=4),
    )
    first.start()
    assert first_read_started.wait(timeout=2)

    second = threading.Thread(
        name="second-update",
        target=lambda: project_manager.update_project(project["id"], chunk_count=9),
    )
    second.start()
    allow_first_read.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    current = project_manager.get_project(project["id"])
    assert current["file_count"] == 4
    assert current["chunk_count"] == 9

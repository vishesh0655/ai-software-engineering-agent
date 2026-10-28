import json
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from backend.settings import settings


PROJECTS_FILE = Path(__file__).resolve().parent.parent / "data" / "projects.json"
CHROMA_DATA_PATH = Path(settings.chroma_path).resolve()
_PROJECTS_LOCK = threading.RLock()
_REPLACE_ATTEMPTS = 5


class InvalidProjectPathError(ValueError):
    pass


class ProjectAlreadyExistsError(ValueError):
    pass


class ProjectNotFoundError(LookupError):
    pass


def normalize_project_path(project_path: str | Path) -> Path:
    try:
        path = Path(project_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidProjectPathError("Project path does not exist or cannot be accessed") from exc
    if not path.is_dir():
        raise InvalidProjectPathError("Project path must be a directory")
    try:
        path.relative_to(CHROMA_DATA_PATH)
    except ValueError:
        return path
    raise InvalidProjectPathError("The ChromaDB data directory cannot be indexed as a project")


def project_id_for_path(path: Path) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", path.name.lower()).strip("_")
    if not name:
        name = "project"
    return name[:55].rstrip("_-") or "project"


def _read_projects() -> dict[str, dict[str, Any]]:
    with _PROJECTS_LOCK:
        if not PROJECTS_FILE.exists():
            return {}
        try:
            data = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read project metadata at {PROJECTS_FILE}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Project metadata file must contain a JSON object")
        return data


def _write_projects(projects: dict[str, dict[str, Any]]) -> None:
    with _PROJECTS_LOCK:
        PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=PROJECTS_FILE.parent,
                prefix="projects-", suffix=".tmp", delete=False,
            ) as temporary_file:
                json.dump(projects, temporary_file, indent=2)
                temporary_file.write("\n")
                temporary_path = Path(temporary_file.name)
            for attempt in range(_REPLACE_ATTEMPTS):
                try:
                    temporary_path.replace(PROJECTS_FILE)
                    break
                except PermissionError:
                    if attempt == _REPLACE_ATTEMPTS - 1:
                        raise
                    time.sleep(0.05 * (2 ** attempt))
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def register_project(project_path: str | Path) -> dict[str, Any]:
    path = normalize_project_path(project_path)
    project_id = project_id_for_path(path)
    with _PROJECTS_LOCK:
        projects = _read_projects()
        existing = projects.get(project_id)
        normalized = str(path).replace("\\", "/")
        if existing:
            if existing.get("path") != normalized:
                raise ProjectAlreadyExistsError(
                    f"Project ID '{project_id}' is already registered for another path"
                )
            return existing

        project = {
            "id": project_id,
            "name": path.name,
            "path": normalized,
            "status": "registered",
            "file_count": 0,
            "chunk_count": 0,
        }
        projects[project_id] = project
        _write_projects(projects)
        return project


def get_project(project_id: str) -> dict[str, Any] | None:
    with _PROJECTS_LOCK:
        return _read_projects().get(project_id)


def list_projects() -> list[dict[str, Any]]:
    with _PROJECTS_LOCK:
        projects = _read_projects()
        return sorted(projects.values(), key=lambda project: project["id"])


def update_project(project_id: str, **updates: Any) -> dict[str, Any]:
    with _PROJECTS_LOCK:
        projects = _read_projects()
        project = projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project '{project_id}' was not found")
        project.update(updates)
        projects[project_id] = project
        _write_projects(projects)
        return project


def delete_project(project_id: str) -> dict[str, Any]:
    if not isinstance(project_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,54}", project_id):
        raise ProjectNotFoundError(f"Project '{project_id}' was not found")
    with _PROJECTS_LOCK:
        projects = _read_projects()
        project = projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project '{project_id}' was not found")

        # Remove the isolated collection before metadata so a storage failure
        # never leaves a registered project pointing at missing vectors.
        from backend.vector_store import VectorStore

        VectorStore.delete_project_collection(project_id)
        projects.pop(project_id)
        _write_projects(projects)
        return project

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.ingestion import get_project_files
from backend.tools import ToolError, ToolResult, failure_result, success_result
from backend.tools.common import project_root


class InspectProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, description="ID of a registered project")


LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".cpp": "cpp", ".c": "c", ".html": "html", ".css": "css",
    ".md": "markdown", ".json": "json", ".sql": "sql",
}


def inspect_project(project_id: str) -> ToolResult:
    try:
        validated = InspectProjectInput(project_id=project_id)
        project, root = project_root(validated.project_id)
        files = get_project_files(root)
        languages: dict[str, int] = {}
        directories: set[str] = set()
        for file in files:
            language = LANGUAGE_BY_EXTENSION.get(file.suffix.lower(), "other")
            languages[language] = languages.get(language, 0) + 1
            parent = file.parent.relative_to(root).as_posix()
            if parent != ".":
                directories.add(parent)

        data = {
            "project_id": project["id"],
            "name": project["name"],
            "path": project["path"],
            "status": project["status"],
            "file_count": project.get("file_count", 0),
            "chunk_count": project.get("chunk_count", 0),
            "languages": dict(sorted(languages.items())),
            "directories": sorted(directories)[:50],
        }
        return success_result("inspect_project", data)
    except ToolError as exc:
        return failure_result("inspect_project", str(exc))
    except ValidationError as exc:
        return failure_result("inspect_project", exc.errors()[0]["msg"])
    except (OSError, ValueError):
        return failure_result("inspect_project", "Unable to inspect project files")
    except Exception:
        return failure_result("inspect_project", "Project inspection failed")


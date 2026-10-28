from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.retrieval import retrieve
from backend.tools import ToolError, ToolResult, failure_result, success_result
from backend.tools.common import project_root, redact_secrets, relative_file, resolve_project_file


class SearchCodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, description="ID of a registered project")
    query: str = Field(min_length=1, description="Non-empty semantic code search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results to return (1–20)")

    @field_validator("project_id", "query")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


def search_code(project_id: str, query: str, max_results: int = 5) -> ToolResult:
    try:
        _, root = project_root(project_id)
        validated = SearchCodeInput(
            project_id=project_id, query=query, max_results=max_results
        )
        retrieved = retrieve(
            validated.query,
            n_results=validated.max_results,
            project_id=validated.project_id,
        )
        results = []
        for item in retrieved:
            metadata = item.get("metadata", {})
            source_path = metadata.get("file", "")
            try:
                source = resolve_project_file(root, source_path)
            except ToolError:
                # Ignore stale or malformed metadata from outside this project.
                continue
            results.append({
                "file": relative_file(root, source),
                "symbol": metadata.get("symbol", ""),
                "type": metadata.get("type", ""),
                "start_line": metadata.get("start_line"),
                "end_line": metadata.get("end_line"),
                "language": metadata.get("language", ""),
                "parent": metadata.get("parent", ""),
                "content": redact_secrets(item.get("content", "")),
            })
        return success_result("search_code", {"results": results})
    except ToolError as exc:
        return failure_result("search_code", str(exc))
    except ValidationError as exc:
        return failure_result("search_code", exc.errors()[0]["msg"])
    except Exception:
        return failure_result("search_code", "Code search failed")


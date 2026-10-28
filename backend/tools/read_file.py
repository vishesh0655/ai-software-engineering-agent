from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.tools import ToolError, ToolResult, failure_result, success_result
from backend.tools.common import project_root, redact_secrets, relative_file, resolve_project_file


MAX_RETURNED_LINES = 300


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, description="ID of a registered project")
    file_path: str = Field(min_length=1, description="Relative path, or absolute path inside the registered project")
    start_line: int | None = Field(default=None, ge=1, description="Optional 1-based first line")
    end_line: int | None = Field(default=None, ge=1, description="Optional inclusive last line; at most 300 lines are returned")


def read_file(
    project_id: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ToolResult:
    try:
        validated = ReadFileInput(
            project_id=project_id, file_path=file_path,
            start_line=start_line, end_line=end_line,
        )
        _, root = project_root(validated.project_id)
        target = resolve_project_file(root, validated.file_path)
        if not target.is_file():
            return failure_result("read_file", "Path is not a file")

        first_line = validated.start_line or 1
        last_line = validated.end_line or first_line + MAX_RETURNED_LINES - 1
        if last_line < first_line:
            return failure_result("read_file", "end_line must be greater than or equal to start_line")
        if last_line - first_line + 1 > MAX_RETURNED_LINES:
            return failure_result("read_file", f"A maximum of {MAX_RETURNED_LINES} lines can be returned")

        selected: list[str] = []
        last_read_line = 0
        with target.open("r", encoding="utf-8", errors="strict") as source:
            for number, line in enumerate(source, start=1):
                last_read_line = number
                if number < first_line:
                    continue
                if number > last_line:
                    break
                selected.append(line)
        if first_line > last_read_line:
            return failure_result("read_file", "start_line is beyond the end of the file")

        content = "".join(selected)
        return success_result("read_file", {
            "file": relative_file(root, target),
            "start_line": first_line,
            "end_line": first_line + max(len(selected), 1) - 1,
            "content": redact_secrets(content),
        })
    except ToolError as exc:
        return failure_result("read_file", str(exc))
    except ValidationError as exc:
        return failure_result("read_file", exc.errors()[0]["msg"])
    except FileNotFoundError:
        return failure_result("read_file", "File not found")
    except UnicodeDecodeError:
        return failure_result("read_file", "File is not valid UTF-8 text")
    except (OSError, RuntimeError):
        return failure_result("read_file", "Unable to read file")
    except Exception:
        return failure_result("read_file", "Invalid file request")


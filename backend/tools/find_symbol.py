import ast
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.ingestion import get_project_files
from backend.tools import ToolError, ToolResult, failure_result, success_result
from backend.tools.common import project_root, relative_file, resolve_project_file


class FindSymbolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, description="ID of a registered project")
    symbol: str = Field(min_length=1, description="Exact function, method, class, or declaration name")
    file_path: str | None = Field(default=None, description="Optional file path inside the registered project")

    @field_validator("project_id", "symbol")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


def _python_matches(source: str, symbol: str, filename: str) -> list[dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _text_matches(source, symbol, filename)

    matches = []

    def visit(node: ast.AST, parent: str | None = None) -> None:
        if isinstance(node, ast.ClassDef):
            if node.name == symbol:
                matches.append({
                    "file": filename, "symbol": node.name, "type": "class",
                    "parent": None, "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                })
            for child in node.body:
                visit(child, node.name)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                matches.append({
                    "file": filename, "symbol": node.name,
                    "type": "method" if parent else "function",
                    "parent": parent, "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                })
            for child in node.body:
                visit(child, None)

    for node in tree.body:
        visit(node)
    return matches


def _text_matches(source: str, symbol: str, filename: str) -> list[dict]:
    name = re.escape(symbol)
    declaration = re.compile(
        rf"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|enum|struct|def|fn|type)\s+{name}\b"
        rf"|^\s*(?:export\s+)?(?:const|let|var)\s+{name}\s*[:=]",
        re.MULTILINE,
    )
    matches = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if declaration.search(line):
            matches.append({
                "file": filename, "symbol": symbol, "type": "text_symbol",
                "parent": None, "start_line": line_number,
                "end_line": line_number,
            })
    return matches


def find_symbol(
    project_id: str,
    symbol: str,
    file_path: str | None = None,
) -> ToolResult:
    try:
        validated = FindSymbolInput(
            project_id=project_id, symbol=symbol, file_path=file_path
        )
        _, root = project_root(validated.project_id)
        if validated.file_path:
            files = [resolve_project_file(root, validated.file_path)]
        else:
            files = get_project_files(root)

        matches = []
        for path in files:
            try:
                resolved = resolve_project_file(root, str(path))
                source = resolved.read_text(encoding="utf-8", errors="strict")
            except ToolError:
                continue
            filename = relative_file(root, resolved)
            if resolved.suffix.lower() == ".py":
                matches.extend(_python_matches(source, validated.symbol, filename))
            else:
                matches.extend(_text_matches(source, validated.symbol, filename))
        return success_result("find_symbol", {"matches": matches})
    except ToolError as exc:
        return failure_result("find_symbol", str(exc))
    except ValidationError as exc:
        return failure_result("find_symbol", exc.errors()[0]["msg"])
    except (OSError, UnicodeDecodeError):
        return failure_result("find_symbol", "Unable to inspect one or more project files")
    except Exception:
        return failure_result("find_symbol", "Symbol search failed")


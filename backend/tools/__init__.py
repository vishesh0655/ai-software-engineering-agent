from typing import Any, TypedDict


class ToolResult(TypedDict):
    success: bool
    tool: str
    data: Any
    error: str | None


class ToolError(Exception):
    pass


def success_result(tool: str, data: Any) -> ToolResult:
    return {"success": True, "tool": tool, "data": data, "error": None}


def failure_result(tool: str, error: str) -> ToolResult:
    return {"success": False, "tool": tool, "data": None, "error": error}


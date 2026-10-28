from typing import Any, Callable

from backend.tools.find_symbol import FindSymbolInput, find_symbol
from backend.tools.inspect_project import InspectProjectInput, inspect_project
from backend.tools.read_file import ReadFileInput, read_file
from backend.tools.run_tests import RunTestsInput, run_tests
from backend.tools.search_code import SearchCodeInput, search_code


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "search_code": search_code,
    "read_file": read_file,
    "find_symbol": find_symbol,
    "inspect_project": inspect_project,
    "run_tests": run_tests,
}

TOOL_SCHEMAS = {
    "search_code": SearchCodeInput,
    "read_file": ReadFileInput,
    "find_symbol": FindSymbolInput,
    "inspect_project": InspectProjectInput,
    "run_tests": RunTestsInput,
}

TOOL_DESCRIPTIONS = {
    "search_code": "Semantically search code chunks in one registered project.",
    "read_file": "Read a bounded line range from a file inside a registered project.",
    "find_symbol": "Find Python or basic text declarations in a registered project.",
    "inspect_project": "Summarize registered project metadata and discovered source files.",
    "run_tests": "Run the registered project's pytest suite with a fixed command and timeout.",
}


def get_tool(name: str) -> Callable[..., dict[str, Any]] | None:
    return TOOLS.get(name)


def get_tool_schema(name: str):
    return TOOL_SCHEMAS.get(name)


def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "input_schema": TOOL_SCHEMAS[name].model_json_schema(),
        }
        for name in TOOLS
    ]

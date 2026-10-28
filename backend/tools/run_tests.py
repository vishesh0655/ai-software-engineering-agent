import importlib.util
import re
import subprocess
import sys
import time

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.ingestion import get_project_files
from backend.tools import ToolError, ToolResult, failure_result, success_result
from backend.tools.common import project_root, resolve_project_file


TEST_TIMEOUT_SECONDS = 120
SUMMARY_COUNTS = {
    "passed_count": re.compile(r"\b(\d+)\s+passed\b"),
    "failed_count": re.compile(r"\b(\d+)\s+failed\b"),
    "skipped_count": re.compile(r"\b(\d+)\s+skipped\b"),
}


def _pytest_summary_counts(output: str) -> dict[str, int]:
    return {
        name: int(match.group(1)) if (match := pattern.search(output)) else 0
        for name, pattern in SUMMARY_COUNTS.items()
    }


class RunTestsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, description="ID of a registered project")


def run_tests(project_id: str) -> ToolResult:
    try:
        validated = RunTestsInput(project_id=project_id)
        _, root = project_root(validated.project_id)
        test_files = []
        for path in get_project_files(root):
            if not (path.name.startswith("test_") or path.name.endswith("_test.py")):
                continue
            try:
                resolve_project_file(root, str(path))
            except Exception:
                continue
            test_files.append(path)
        if not test_files:
            return failure_result("run_tests", "No pytest-style test files were found")
        if importlib.util.find_spec("pytest") is None:
            return failure_result("run_tests", "pytest is not installed in the API environment")

        started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=str(root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TEST_TIMEOUT_SECONDS,
                check=False,
            )
            exit_code = completed.returncode
            output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            output = "\n".join(part for part in (stdout, stderr, f"Tests exceeded {TEST_TIMEOUT_SECONDS} seconds") if part).strip()
        duration = round(time.monotonic() - started, 3)
        counts = _pytest_summary_counts(output)
        return success_result("run_tests", {
            "passed": exit_code == 0,
            "exit_code": exit_code,
            **counts,
            "output": output,
            "duration_seconds": duration,
        })
    except ToolError as exc:
        return failure_result("run_tests", str(exc))
    except ValidationError as exc:
        return failure_result("run_tests", exc.errors()[0]["msg"])
    except (OSError, subprocess.SubprocessError):
        return failure_result("run_tests", "Unable to start the fixed pytest runner")
    except Exception:
        return failure_result("run_tests", "Test execution failed")


"""Read-only Git inspection using fixed argument lists (never a shell)."""
import subprocess
from typing import Any

from backend.tools.common import project_root, redact_secrets


class GitInspectionError(RuntimeError):
    pass


def inspect_git(project_id: str) -> dict[str, Any]:
    _, root = project_root(project_id)

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "-c", "core.fsmonitor=false", "-c", "core.pager=cat", *arguments],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=10, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitInspectionError("Git inspection is unavailable or timed out") from exc
        if result.returncode:
            raise GitInspectionError("Git inspection failed")
        return result.stdout

    try:
        inside = git("rev-parse", "--is-inside-work-tree").strip() == "true"
    except GitInspectionError:
        return {"is_git_repository": False, "branch": None, "changed_files": [], "diff": "", "diff_truncated": False}
    if not inside:
        return {"is_git_repository": False, "branch": None, "changed_files": [], "diff": "", "diff_truncated": False}
    branch = git("branch", "--show-current").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    files = []
    for line in status.splitlines():
        if len(line) >= 4:
            files.append({"status": line[:2].strip() or "modified", "path": line[3:]})
    diff = git("diff", "HEAD", "--no-ext-diff", "--no-textconv", "--no-color", "--", ".")
    return {
        "is_git_repository": inside,
        "branch": branch or None,
        "changed_files": files,
        "diff": redact_secrets(diff)[:100_000],
        "diff_truncated": len(diff) > 100_000,
    }

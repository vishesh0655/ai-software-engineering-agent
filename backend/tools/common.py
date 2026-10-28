import re
from pathlib import Path
from typing import Any

from backend.project_manager import get_project
from backend.tools import ToolError


SENSITIVE_PARTS = {
    ".aws", ".git", ".ssh", ".venv", "venv", "node_modules",
    "__pycache__", "chroma",
}
SENSITIVE_NAMES = {"id_rsa", "id_ed25519", "credentials", "secrets"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks"}
SECRET_NAME = (
    r"(?:\w*(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|credential)\w*"
    r"|(?:access|refresh|auth|jwt|id)[_-]?token|token)"
)
SECRET_ASSIGNMENT = re.compile(
    rf"(?i)(\b{SECRET_NAME}\s*=\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
SECRET_JSON_VALUE = re.compile(
    rf"(?i)([\"']{SECRET_NAME}[\"']\s*:\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
BEARER_TOKEN = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
AWS_ACCESS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")


def project_root(project_id: str) -> tuple[dict[str, Any], Path]:
    project = get_project(project_id)
    if project is None:
        raise ToolError("Project not found")
    try:
        root = Path(project["path"]).resolve(strict=True)
    except (KeyError, OSError, RuntimeError) as exc:
        raise ToolError("Registered project directory is unavailable") from exc
    if not root.is_dir():
        raise ToolError("Registered project directory is unavailable")
    return project, root


def resolve_project_file(root: Path, file_path: str, must_exist: bool = True) -> Path:
    if not file_path or not file_path.strip():
        raise ToolError("file_path must not be empty")
    supplied = Path(file_path)
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
        if must_exist and not resolved.exists():
            raise FileNotFoundError(str(resolved))
    except FileNotFoundError as exc:
        raise ToolError("File not found") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolError("File path must stay inside the registered project") from exc
    if any(part.lower() in SENSITIVE_PARTS for part in resolved.relative_to(root).parts):
        raise ToolError("Access to internal or sensitive project paths is denied")
    if resolved.name.lower() in SENSITIVE_NAMES or resolved.suffix.lower() in SENSITIVE_SUFFIXES:
        raise ToolError("Access to sensitive files is denied")
    if resolved.name.lower().startswith(".env"):
        raise ToolError("Access to environment files is denied")
    return resolved


def relative_file(root: Path, file_path: Path) -> str:
    return file_path.relative_to(root).as_posix()


def redact_secrets(content: str) -> str:
    content = SECRET_ASSIGNMENT.sub(
        lambda match: f'{match.group(1)}{match.group("quote")}[REDACTED]{match.group("quote")}',
        content,
    )
    content = SECRET_JSON_VALUE.sub(
        lambda match: f'{match.group(1)}{match.group("quote")}[REDACTED]{match.group("quote")}', content
    )
    content = BEARER_TOKEN.sub(r"\1[REDACTED]", content)
    return AWS_ACCESS_KEY.sub("[REDACTED]", content)


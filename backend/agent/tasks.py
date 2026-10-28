import difflib
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.agent.agent import AgentError, run_agent
from backend.agent.protocol import ProtocolError
from backend.llm import ModelGenerationError, generate
from backend.project_manager import get_project
from backend.tools import ToolError
from backend.tools.common import project_root, redact_secrets, resolve_project_file
from backend.tools.registry import get_tool


TASKS_FILE = Path(__file__).resolve().parents[2] / "data" / "tasks.json"
BACKUP_ROOT = Path(__file__).resolve().parents[2] / "data" / "task_backups"
MAX_FILES_PER_PROPOSAL = 10
MAX_FILE_BYTES = 512_000
MAX_PROPOSAL_BYTES = 2_000_000
MAX_PROPOSAL_RESPONSE_CHARS = 2_500_000
MAX_TASK_REQUEST_CHARS = 4_000
MAX_FIX_ATTEMPTS = 3
MAX_TEST_OUTPUT_CHARS = 12_000
_TASK_LOCK = threading.RLock()
logger = logging.getLogger("backend.agent.tasks")


class TaskError(RuntimeError):
    """A task cannot perform the requested state transition."""


class ProposedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ProposedChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: list[ProposedFile] = Field(min_length=1, max_length=MAX_FILES_PER_PROPOSAL)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_tasks() -> dict[str, dict[str, Any]]:
    if not TASKS_FILE.exists():
        return {}
    try:
        tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskError("Task state could not be read") from exc
    if not isinstance(tasks, dict):
        raise TaskError("Task state is malformed")
    return tasks


def _save_tasks(tasks: dict[str, dict[str, Any]]) -> None:
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=TASKS_FILE.parent,
            prefix="tasks-", suffix=".tmp", delete=False,
        ) as stream:
            json.dump(tasks, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(TASKS_FILE)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _get_task(tasks: dict[str, dict[str, Any]], task_id: str) -> dict[str, Any]:
    task = tasks.get(task_id)
    if task is None:
        raise TaskError("Task was not found")
    return task


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in task.items() if key not in {"proposal", "applications", "project_path"}}
    proposal = task.get("proposal")
    if proposal:
        public["proposed_changes"] = [
            {
                "path": change["path"],
                "reason": change["reason"],
                "diff": redact_secrets(change["diff"]),
                "approval_status": proposal["approval_status"],
            }
            for change in proposal["changes"]
        ]
    else:
        public["proposed_changes"] = []
    return public


def _task_failure_diagnostic(stage: str, exc: Exception, task: dict[str, Any]) -> dict[str, str]:
    if stage == "proposal":
        return {
            "stage": stage,
            "code": "PROPOSAL_GENERATION_FAILED",
            "message": "Agent investigation completed, but Buildmate could not create a valid structured change proposal. Review task evidence and backend diagnostics.",
        }

    if isinstance(exc, AgentError):
        cause = exc.__cause__
        cause_text = str(cause).lower() if cause else ""
        error_text = str(exc).lower()
        if isinstance(cause, ModelGenerationError):
            code, message = "MODEL_EMPTY_RESPONSE", "Ollama completed the request but returned no usable text. Check the model output and retry the task."
        elif isinstance(cause, ProtocolError):
            if "empty or whitespace" in cause_text:
                code, message = "AGENT_EMPTY_RESPONSE", "The model returned an empty response instead of an agent action. Retry the task and check model output if this repeats."
            elif "valid json" in cause_text:
                code, message = "AGENT_INVALID_JSON", "The agent returned invalid JSON repeatedly and could not continue safely."
            elif "unknown action" in cause_text:
                code, message = "AGENT_INVALID_TOOL", "The agent requested a tool that Buildmate does not allow."
            elif "malformed tool action" in cause_text:
                code, message = "AGENT_INVALID_TOOL_ACTION", "The agent's tool action was missing a required field or included an unsupported field."
            elif "arguments failed validation" in cause_text:
                code, message = "AGENT_INVALID_TOOL_ARGUMENTS", "The agent's tool request did not match the allowed input format."
            elif "final" in cause_text:
                code, message = "AGENT_INVALID_RESPONSE", "The agent returned an incomplete or invalid final response."
            else:
                code, message = "AGENT_INVALID_ACTION", "The agent returned an invalid action repeatedly and could not continue safely."
        elif cause is not None and any(term in cause_text for term in ("timed out", "timeout", "readtimeout", "connecttimeout")):
            code, message = "MODEL_TIMEOUT", "The agent model service timed out. Check Ollama and retry the task."
        elif cause is not None and "model is unavailable" in error_text:
            code, message = "MODEL_UNAVAILABLE", "The agent could not reach the model service. Check Ollama and retry the task."
        elif "final answer lacked tool evidence" in error_text:
            code, message = "AGENT_EVIDENCE_MISSING", "The agent did not gather registered tool evidence before finishing."
        elif "maximum number of tool calls" in error_text:
            code, message = "AGENT_TOOL_LIMIT", "The agent reached its allowed tool-call limit before finishing."
        elif "maximum number of iterations" in error_text:
            code, message = "AGENT_ITERATION_LIMIT", "The agent reached its investigation limit before producing a final answer."
        else:
            failed_tools = [call.get("tool", "registered tool") for call in task.get("tool_calls", []) if not call.get("success")]
            if failed_tools:
                code, message = "AGENT_TOOL_FAILURE", f"The agent could not complete after a {failed_tools[-1]} tool call failed. Review agent activity and retry."
            else:
                code, message = "AGENT_INVESTIGATION_FAILED", "Agent investigation failed. Review task evidence and backend diagnostics."
        diagnostic = {"stage": stage, "code": code, "message": message}
        if cause is not None:
            root_cause = cause
            while root_cause.__cause__ is not None:
                root_cause = root_cause.__cause__
            diagnostic["cause_type"] = type(cause).__name__
            diagnostic["root_cause_type"] = type(root_cause).__name__
            if isinstance(cause, ProtocolError):
                diagnostic["cause_message"] = str(cause)[:200]
                if cause.response_length is not None:
                    diagnostic["response_length"] = cause.response_length
                if cause.response_start_kind is not None:
                    diagnostic["response_start_kind"] = cause.response_start_kind
        return diagnostic

    return {
        "stage": stage,
        "code": "AGENT_INVESTIGATION_FAILED",
        "message": "Agent investigation failed unexpectedly. Review task evidence and backend diagnostics.",
    }


def get_task(task_id: str) -> dict[str, Any]:
    # Atomic file replacement means readers can safely poll without waiting for
    # a long running registered test command held by a task transition.
    return _public_task(_get_task(_load_tasks(), task_id))


def list_tasks(project_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    tasks = _load_tasks().values()
    selected = [task for task in tasks if project_id is None or task.get("project_id") == project_id]
    selected.sort(key=lambda task: task.get("updated_at", ""), reverse=True)
    return [_public_task(task) for task in selected[:max(1, min(limit, 100))]]


def _infer_mode(request: str, requested_mode: str) -> Literal["investigate", "change", "review"]:
    if requested_mode == "investigate":
        return "investigate"
    if requested_mode == "change":
        return "change"
    if requested_mode == "review":
        return "review"
    lowered = request.lower()
    wants_change = re.search(r"\b(fix|repair|change|modify|implement|add|remove|update|refactor|solve)\b", lowered)
    wants_review = re.search(r"\b(review|audit|code review|security review)\b", lowered)
    if wants_change:
        return "change"
    if wants_review:
        return "review"
    return "investigate"


def _investigation_request(normalized: str, selected_mode: str) -> str:
    if selected_mode == "review":
        return (
            "Perform a code review of this project for bugs, security problems, maintainability, "
            "performance, and testing gaps. This is review-only: do not generate code, patches, or diffs; "
            "do not say or imply that files were modified. Do not propose changes. Report only findings "
            "with file/symbol evidence and distinguish confirmed issues from concerns. If there are no "
            "confirmed findings, say so clearly. User request: " + normalized
        )
    return normalized


def start_task(project_id: str, request: str, mode: str = "auto") -> dict[str, Any]:
    project = get_project(project_id)
    if project is None:
        raise TaskError("Project was not found")
    normalized = request.strip()
    if not normalized:
        raise TaskError("Request must not be empty")
    if len(normalized) > MAX_TASK_REQUEST_CHARS:
        raise TaskError("Request exceeds the configured length limit")
    selected_mode = _infer_mode(normalized, mode)
    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id, "project_id": project_id, "request": normalized,
        "project_path": project["path"],
        "mode": selected_mode, "status": "investigating", "answer": None,
        "diagnosis": None, "tool_calls": [], "iterations": 0,
        "error": None,
        "actions": [{"at": _now(), "action": "investigation_started", "summary": "Agent is investigating the request"}],
        "proposal": None,
        "approval_status": "not_required" if selected_mode != "change" else "not_proposed",
        "test_results": [], "attempts": 0, "applications": [], "result": None,
        "created_at": _now(), "updated_at": _now(),
    }
    with _TASK_LOCK:
        tasks = _load_tasks()
        tasks[task_id] = task
        _save_tasks(tasks)
    return _public_task(task)


def _record_tool_call(task_id: str, call: dict[str, Any]) -> None:
    with _TASK_LOCK:
        tasks = _load_tasks()
        task = _get_task(tasks, task_id)
        event = {**call, "at": _now()}
        task["tool_calls"].append(event)
        task["actions"].append({
            "at": _now(), "action": "tool_call",
            "summary": f"{call['tool'].replace('_', ' ').capitalize()} {'completed' if call['success'] else 'failed'}",
        })
        task["updated_at"] = _now()
        _save_tasks(tasks)
        logger.info(
            "Agent tool recorded task_id=%s project_id=%s mode=%s iteration=%s tool=%s success=%s",
            task_id, task.get("project_id"), task.get("mode"), call.get("iteration"), call.get("tool"), call.get("success"),
        )


def _record_iteration(task_id: str, iteration: int) -> None:
    with _TASK_LOCK:
        tasks = _load_tasks()
        task = _get_task(tasks, task_id)
        task["iterations"] = iteration
        task["updated_at"] = _now()
        _save_tasks(tasks)


def process_task(task_id: str) -> dict[str, Any]:
    with _TASK_LOCK:
        tasks = _load_tasks()
        task = _get_task(tasks, task_id)
        if task["status"] != "investigating":
            return _public_task(task)
        selected_mode = task["mode"]
        request = _investigation_request(task["request"], selected_mode)
        project_id = task["project_id"]
    stage = "investigation"
    logger.info(
        "Agent task started task_id=%s project_id=%s mode=%s",
        task_id, project_id, selected_mode,
    )
    try:
        investigation = run_agent(
            project_id, request,
            on_tool_call=lambda call: _record_tool_call(task_id, call),
            on_iteration=lambda iteration: _record_iteration(task_id, iteration),
        )
        logger.info(
            "Agent task investigation completed task_id=%s project_id=%s mode=%s iterations=%s tool_count=%s",
            task_id, project_id, selected_mode, investigation.get("iterations", 0), len(investigation.get("tool_calls", [])),
        )
        with _TASK_LOCK:
            tasks = _load_tasks()
            task = _get_task(tasks, task_id)
            task["answer"] = investigation["answer"]
            task["diagnosis"] = investigation["answer"] if selected_mode == "change" else None
            task["iterations"] = investigation.get("iterations", 0)
            task["actions"].append({"at": _now(), "action": "investigated", "summary": investigation["answer"]})
            task["status"] = "investigated"
            task["updated_at"] = _now()
            if selected_mode == "change":
                task["status"] = "generating_proposal"
            _save_tasks(tasks)
        if selected_mode == "change":
            stage = "proposal"
            proposal = _make_proposal(task, diagnosis=task["diagnosis"])
            with _TASK_LOCK:
                tasks = _load_tasks()
                task = _get_task(tasks, task_id)
                task["proposal"] = proposal
                task["status"] = "awaiting_approval"
                task["error"] = None
                task["approval_status"] = "pending"
                task["actions"].append({"at": _now(), "action": "change_proposed", "summary": "A diff is ready for review"})
                task["updated_at"] = _now()
                _save_tasks(tasks)
        return get_task(task_id)
    except Exception as exc:
        with _TASK_LOCK:
            task = _get_task(_load_tasks(), task_id)
        diagnostic = _task_failure_diagnostic(stage, exc, task)
        logger.exception(
            "Background task failed task_id=%s project_id=%s mode=%s stage=%s exception_type=%s error=%s iterations=%s tool_count=%s last_tool=%s last_tool_success=%s diagnostic_code=%s",
            task_id, project_id, selected_mode, stage, type(exc).__name__,
            redact_secrets(str(exc))[:500], task.get("iterations", 0),
            len(task.get("tool_calls", [])),
            task.get("tool_calls", [{}])[-1].get("tool") if task.get("tool_calls") else None,
            task.get("tool_calls", [{}])[-1].get("success") if task.get("tool_calls") else None,
            diagnostic["code"],
        )
        with _TASK_LOCK:
            tasks = _load_tasks()
            task = _get_task(tasks, task_id)
            task["status"] = "failed"
            task["error"] = diagnostic
            task["result"] = diagnostic["message"]
            task["actions"].append({"at": _now(), "action": "failed", "summary": task["result"]})
            task["updated_at"] = _now()
            _save_tasks(tasks)
        return get_task(task_id)


def _safe_target(root: Path, relative_path: str) -> Path:
    if not relative_path.strip() or Path(relative_path).is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise TaskError("Proposed file path must be relative to the selected project")
    if ".." in relative_path.replace("\\", "/").split("/") or ":" in relative_path:
        raise TaskError("Proposed file path must stay inside the selected project")
    try:
        target = resolve_project_file(root, relative_path, must_exist=False)
        parent = target.parent.resolve(strict=True)
        parent.relative_to(root)
    except (OSError, ValueError, RuntimeError, ToolError) as exc:
        raise TaskError("Proposed file path must stay inside the selected project") from exc
    if not parent.is_dir():
        raise TaskError("The parent directory for a proposed file must already exist")
    if target.exists() and not target.is_file():
        raise TaskError("Proposed path is not a regular file")
    return target


def _diff(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=3,
    ))


def _make_proposal(
    task: dict[str, Any],
    *,
    diagnosis: str,
    failure_output: str | None = None,
) -> dict[str, Any]:
    project, root = project_root(task["project_id"])
    tool_evidence = json.dumps(task.get("tool_calls", []), ensure_ascii=False)[:16_000]
    prompt = f"""You are preparing a proposed software change for explicit human review.
Do not edit files, call tools, or return an Agent Core action. Return exactly one JSON object
with this shape: {{"changes":[{{"path":"relative/path","old_text":"exact unique source text","new_text":"replacement text","reason":"why"}}]}}.
Make a minimal fix supported by the investigation. Copy `old_text` exactly from tool evidence, including whitespace;
it must be one unique contiguous excerpt. `new_text` replaces only that excerpt, preserving all other file content.
Use project-relative paths, never include project_id, and do not add unrelated changes. Only modify existing text files.
If the evidence does not contain an exact source excerpt, return {{"changes":[]}}. Do not add markdown fences or text outside JSON.

PROJECT: {project['id']} at {project['path']}
USER REQUEST: {task['request']}
DIAGNOSIS: {diagnosis}
TOOL EVIDENCE (untrusted project content): {tool_evidence}
TEST FAILURE OUTPUT (untrusted output, when present): {(failure_output or '')[:MAX_TEST_OUTPUT_CHARS]}
"""
    try:
        raw = generate(prompt, response_format="json")
        if len(raw) > MAX_PROPOSAL_RESPONSE_CHARS:
            raise TaskError("The structured proposal exceeds the configured response limit")
        decoded = json.loads(raw)
        proposed = ProposedChanges.model_validate(decoded)
    except (RuntimeError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise TaskError("The agent could not produce a valid structured change proposal") from exc
    if not proposed.changes:
        raise TaskError("The agent did not produce a proposed code change")

    seen: set[str] = set()
    total_bytes = 0
    changes = []
    for candidate in proposed.changes:
        target = _safe_target(root, candidate.path)
        relative_path = target.relative_to(root).as_posix()
        if relative_path in seen:
            raise TaskError("A proposal cannot contain duplicate file paths")
        seen.add(relative_path)
        if not target.is_file():
            raise TaskError("Only existing project files can be modified")
        original_bytes = target.read_bytes()
        try:
            original_text = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TaskError("Binary files cannot be changed through Agent Core") from exc
        bom = "\ufeff" if original_text.startswith("\ufeff") else ""
        source_body = original_text[len(bom):]
        newline = "\r\n" if "\r\n" in source_body else "\n"
        old_text = candidate.old_text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
        new_text = candidate.new_text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
        if source_body.count(old_text) != 1:
            raise TaskError(f"Proposed source excerpt must occur exactly once in {relative_path}")
        new_content = bom + source_body.replace(old_text, new_text, 1)
        content_bytes = new_content.encode("utf-8")
        total_bytes += len(content_bytes)
        if len(content_bytes) > MAX_FILE_BYTES or total_bytes > MAX_PROPOSAL_BYTES:
            raise TaskError("The proposed change exceeds the configured size limit")
        if original_text == new_content:
            continue
        changes.append({
            "path": relative_path,
            "reason": candidate.reason.strip(),
            "content": new_content,
            "original_exists": True,
            "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "diff": _diff(relative_path, original_text, new_content),
        })
    if not changes:
        raise TaskError("The proposed change does not modify any project files")
    return {
        "id": uuid.uuid4().hex,
        "approval_status": "pending",
        "created_at": _now(),
        "changes": changes,
    }


def create_task(project_id: str, request: str, mode: str = "auto") -> dict[str, Any]:
    project = get_project(project_id)
    if project is None:
        raise TaskError("Project was not found")
    normalized = request.strip()
    if not normalized:
        raise TaskError("Request must not be empty")
    if len(normalized) > MAX_TASK_REQUEST_CHARS:
        raise TaskError("Request exceeds the configured length limit")
    selected_mode = _infer_mode(normalized, mode)
    investigation_request = _investigation_request(normalized, selected_mode)
    try:
        investigation = run_agent(project_id, investigation_request)
    except RuntimeError as exc:
        raise TaskError(f"Project investigation failed: {exc}") from exc

    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "project_id": project_id,
        "project_path": project["path"],
        "request": normalized,
        "mode": selected_mode,
        "status": "investigated",
        "error": None,
        "answer": investigation["answer"],
        "diagnosis": investigation["answer"] if selected_mode == "change" else None,
        "tool_calls": investigation.get("tool_calls", []),
        "iterations": investigation.get("iterations", 0),
        "actions": [{"at": _now(), "action": "investigated", "summary": investigation["answer"]}],
        "proposal": None,
        "approval_status": "not_required" if selected_mode != "change" else "not_proposed",
        "test_results": [],
        "attempts": 0,
        "applications": [],
        "result": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    if selected_mode == "change":
        task["proposal"] = _make_proposal(task, diagnosis=task["diagnosis"])
        task["status"] = "awaiting_approval"
        task["approval_status"] = "pending"
        task["actions"].append({"at": _now(), "action": "change_proposed", "summary": "A diff is ready for review"})
    with _TASK_LOCK:
        tasks = _load_tasks()
        tasks[task_id] = task
        _save_tasks(tasks)
    return _public_task(task)


def _backup_path(task_id: str, backup_name: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", task_id) or not re.fullmatch(r"[0-9]+\.bak", backup_name):
        raise TaskError("Backup reference is invalid")
    root = BACKUP_ROOT.resolve()
    path = (root / task_id / backup_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TaskError("Backup reference is invalid") from exc
    return path


def _rollback_root(task: dict[str, Any]) -> Path:
    """Resolve the original project for rollback, even after registry removal."""
    project = get_project(task["project_id"])
    original_path = task.get("project_path")
    if project is not None:
        try:
            registered_root = Path(project["path"]).resolve(strict=True)
            if not registered_root.is_dir():
                raise OSError("registered project is not a directory")
            if original_path:
                original_root = Path(original_path).resolve(strict=True)
                if original_root != registered_root:
                    raise TaskError("Project path changed since this task was created; refusing rollback")
            return registered_root
        except (KeyError, OSError, RuntimeError) as exc:
            raise TaskError("Registered project directory is unavailable") from exc
    if not original_path:
        raise TaskError("Project is no longer registered and this task has no saved rollback location")
    try:
        root = Path(original_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TaskError("Original project directory is unavailable; rollback was not started") from exc
    if not root.is_dir():
        raise TaskError("Original project directory is unavailable; rollback was not started")
    return root


def _write_atomic(target: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=".agent-change-", delete=False) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _run_tests(project_id: str) -> dict[str, Any]:
    runner = get_tool("run_tests")
    if runner is None:
        logger.error("Test tool is missing from the registered tool set")
        return {"success": False, "error": "run_tests is not registered"}
    try:
        result = runner(project_id=project_id)
    except Exception as exc:
        logger.exception("Registered test tool raised an exception")
        return {"success": False, "error": f"Test tool failed: {type(exc).__name__}"}
    return result


def _apply_proposal(task: dict[str, Any]) -> dict[str, Any]:
    _, root = project_root(task["project_id"])
    proposal = task["proposal"]
    prepared = []
    for index, change in enumerate(proposal["changes"]):
        target = _safe_target(root, change["path"])
        if target.exists():
            current = target.read_bytes()
            current_hash = hashlib.sha256(current).hexdigest()
            if not change["original_exists"] or current_hash != change["original_sha256"]:
                raise TaskError(f"File changed since proposal approval: {change['path']}")
        else:
            current = None
            if change["original_exists"]:
                raise TaskError(f"File changed since proposal approval: {change['path']}")
        prepared.append((index, change, target, current))

    backup_dir = BACKUP_ROOT / task["task_id"]
    backup_dir.mkdir(parents=True, exist_ok=True)
    application = {"proposal_id": proposal["id"], "files": [], "applied_at": _now()}
    for index, change, target, current in prepared:
        backup_name = f"{len(task['applications']) * MAX_FILES_PER_PROPOSAL + index}.bak"
        backup = _backup_path(task["task_id"], backup_name)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if current is not None:
            backup.write_bytes(current)
        application["files"].append({
            "path": change["path"], "backup_name": backup_name,
            "existed": current is not None,
            "applied_sha256": hashlib.sha256(change["content"].encode("utf-8")).hexdigest(),
        })

    written: list[tuple[Path, bytes | None]] = []
    try:
        for _, change, target, current in prepared:
            _write_atomic(target, change["content"].encode("utf-8"))
            written.append((target, current))
    except Exception as exc:
        rollback_errors = []
        for target, original in reversed(written):
            try:
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    _write_atomic(target, original)
            except OSError:
                rollback_errors.append(target.name)
        if rollback_errors:
            raise TaskError("Applying the proposal failed and automatic restoration was incomplete") from exc
        raise TaskError("Applying the proposal failed; original files were restored") from exc
    return application


def approve_task(task_id: str) -> dict[str, Any]:
    with _TASK_LOCK:
        tasks = _load_tasks()
        task = _get_task(tasks, task_id)
        if task["status"] != "awaiting_approval" or not task.get("proposal"):
            raise TaskError("Task has no pending change proposal to approve")
        task["proposal"]["approval_status"] = "approved"
        task["approval_status"] = "approved"
        task["actions"].append({"at": _now(), "action": "approved", "summary": "Explicit approval received"})
        try:
            application = _apply_proposal(task)
        except Exception as exc:
            logger.warning("Task %s failed while applying its approved proposal: %s", task_id, type(exc).__name__)
            message = str(exc) if isinstance(exc, TaskError) else "Applying the proposal failed before completion"
            task["status"] = "apply_failed"
            task["result"] = message
            task["actions"].append({"at": _now(), "action": "apply_failed", "summary": message})
            task["updated_at"] = _now()
            _save_tasks(tasks)
            return _public_task(task)
        task["applications"].append(application)
        task["attempts"] += 1
        task["status"] = "testing"
        task["updated_at"] = _now()
        _save_tasks(tasks)

        test_result = _run_tests(task["project_id"])
        data = test_result.get("data") if isinstance(test_result, dict) else None
        passed = bool(test_result.get("success") and isinstance(data, dict) and data.get("passed") is True)
        test_record = {
            "attempt": task["attempts"], "passed": passed,
            "success": bool(test_result.get("success")) if isinstance(test_result, dict) else False,
            "exit_code": data.get("exit_code") if isinstance(data, dict) else None,
            "passed_count": data.get("passed_count", 0) if isinstance(data, dict) else 0,
            "failed_count": data.get("failed_count", 0) if isinstance(data, dict) else 0,
            "skipped_count": data.get("skipped_count", 0) if isinstance(data, dict) else 0,
            "output": redact_secrets(str(data.get("output", test_result.get("error", "")) if isinstance(data, dict) else test_result.get("error", "")))[:MAX_TEST_OUTPUT_CHARS],
            "duration_seconds": data.get("duration_seconds") if isinstance(data, dict) else None,
            "at": _now(),
        }
        task["test_results"].append(test_record)
        if not passed:
            logger.warning("Tests failed for approved task %s on attempt %s", task_id, task["attempts"])
        task["actions"].append({
            "at": _now(), "action": "tests_passed" if passed else "tests_failed",
            "summary": "Tests passed" if passed else "Tests failed or could not be run",
        })
        if passed:
            task["status"] = "completed"
            task["result"] = "Approved changes were applied and the registered test tool passed."
            task["proposal"]["approval_status"] = "applied"
        elif task["attempts"] < MAX_FIX_ATTEMPTS:
            task["status"] = "analyzing_test_failure"
            _save_tasks(tasks)
            try:
                diagnosis = run_agent(
                    task["project_id"],
                    "Analyze this approved change's test failure. Identify the likely code issue using the registered tools. "
                    "Give a concise evidence-based diagnosis and do not claim the tests passed. "
                    f"Failure output:\n{test_record['output']}",
                )
                task["diagnosis"] = diagnosis["answer"]
                task["tool_calls"].extend(diagnosis.get("tool_calls", []))
                task["proposal"] = _make_proposal(
                    task, diagnosis=diagnosis["answer"], failure_output=test_record["output"],
                )
                task["proposal"]["approval_status"] = "pending"
                task["status"] = "awaiting_approval"
                task["approval_status"] = "pending"
                task["actions"].append({"at": _now(), "action": "fix_proposed", "summary": "A follow-up proposal is ready for approval"})
            except RuntimeError as exc:
                task["status"] = "tests_failed"
                task["result"] = f"Tests failed; follow-up proposal could not be created: {exc}"
        else:
            task["status"] = "tests_failed"
            task["result"] = "Tests failed; the maximum approved fix attempts was reached."
        task["updated_at"] = _now()
        _save_tasks(tasks)
        return _public_task(task)


def reject_task(task_id: str) -> dict[str, Any]:
    with _TASK_LOCK:
        tasks = _load_tasks()
        task = _get_task(tasks, task_id)
        if task["status"] != "awaiting_approval" or not task.get("proposal"):
            raise TaskError("Task has no pending change proposal to reject")
        task["proposal"]["approval_status"] = "rejected"
        task["approval_status"] = "rejected"
        task["status"] = "rejected"
        task["result"] = "The proposed change was rejected; no files were modified."
        task["actions"].append({"at": _now(), "action": "rejected", "summary": "Explicit rejection received"})
        task["updated_at"] = _now()
        _save_tasks(tasks)
        return _public_task(task)


def rollback_task(task_id: str) -> dict[str, Any]:
    with _TASK_LOCK:
        tasks = _load_tasks()
        task = _get_task(tasks, task_id)
        if not task.get("applications") or task["status"] in {"rolled_back", "rolling_back"}:
            raise TaskError("Task has no applied changes available to roll back")
        root = _rollback_root(task)
        restore_plan = []
        latest: dict[str, str] = {}
        for application in reversed(task["applications"]):
            for item in reversed(application["files"]):
                latest.setdefault(item["path"], item["applied_sha256"])
                target = _safe_target(root, item["path"])
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != latest[item["path"]]:
                    raise TaskError(f"File changed after application; refusing rollback: {item['path']}")
                backup = _backup_path(task_id, item["backup_name"])
                if item["existed"] and not backup.is_file():
                    raise TaskError("A required backup is missing; rollback was not started")
                restore_plan.append((item, target, backup))

        current_bytes = {str(target): target.read_bytes() for _, target, _ in restore_plan}
        restored: list[Path] = []
        task["status"] = "rolling_back"
        _save_tasks(tasks)
        try:
            for item, target, backup in restore_plan:
                if item["existed"]:
                    _write_atomic(target, backup.read_bytes())
                else:
                    target.unlink(missing_ok=True)
                restored.append(target)
        except Exception as exc:
            logger.exception("Rollback failed for task %s", task_id)
            for target in reversed(restored):
                try:
                    _write_atomic(target, current_bytes[str(target)])
                except OSError:
                    pass
            task["status"] = "rollback_failed"
            task["result"] = "Rollback failed; restoration was attempted."
            task["updated_at"] = _now()
            _save_tasks(tasks)
            raise TaskError("Rollback failed; restoration was attempted") from exc
        task["status"] = "rolled_back"
        task["approval_status"] = "rolled_back"
        task["result"] = "Applied changes were rolled back from verified backups."
        task["actions"].append({"at": _now(), "action": "rolled_back", "summary": task["result"]})
        task["updated_at"] = _now()
        _save_tasks(tasks)
        return _public_task(task)

from typing import Any, Literal
import logging

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError

from backend.agent import AgentError, run_agent
from backend.agent.tasks import (
    TaskError,
    approve_task,
    create_task,
    get_task,
    list_tasks,
    process_task,
    start_task,
    reject_task,
    rollback_task,
)
from backend.rag import RAGUnavailableError, ask_codebase
from backend.project_indexer import ChromaIndexError, EmptyProjectError, index_registered_project
from backend.embeddings import EmbeddingError, EmbeddingUnavailableError
from backend.project_manager import (
    InvalidProjectPathError,
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
    delete_project,
    get_project,
    list_projects,
    register_project,
)
from backend.tools.registry import get_tool, get_tool_schema, list_tools
from backend.git_support import GitInspectionError, inspect_git

logger = logging.getLogger("backend.api")


app = FastAPI(
    title="AI Software Engineering Agent",
    description="API for querying a codebase using RAG and a local LLM.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    project_id: str | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value.strip()


class AgentRequest(BaseModel):
    project_id: str
    question: str

    @field_validator("project_id", "question")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty or whitespace")
        return value


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=55)
    request: str = Field(min_length=1, max_length=4000)
    mode: Literal["auto", "investigate", "change", "review"] = "auto"

    @field_validator("project_id", "request")
    @classmethod
    def task_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty or whitespace")
        return value


class Source(BaseModel):
    file: str
    symbol: str
    type: str
    start_line: int
    end_line: int
    language: str | None = None
    parent: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class ProjectIndexRequest(BaseModel):
    project_path: str

    @field_validator("project_path")
    @classmethod
    def project_path_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("project_path must not be empty or whitespace")
        return value.strip()


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Any, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request body"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if request.project_id and get_project(request.project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{request.project_id}' was not found")
    try:
        result = ask_codebase(request.question, project_id=request.project_id)
    except RAGUnavailableError as exc:
        logger.warning("RAG request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (requests.RequestException, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="AI service is unavailable. Check the Ollama URL and configured model.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected RAG failure")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while answering the question.",
        ) from exc

    return AskResponse(**result)


@app.post("/agent")
def agent(request: AgentRequest) -> dict[str, Any]:
    if get_project(request.project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{request.project_id}' was not found")
    try:
        return run_agent(request.project_id, request.question)
    except AgentError as exc:
        logger.exception(
            "Agent API request failed project_id=%s exception_type=%s error=%s",
            request.project_id, type(exc).__name__, str(exc)[:300],
        )
        raise HTTPException(status_code=502, detail="Agent investigation failed. Check the model service and registered project.") from exc


def _task_error(exc: TaskError) -> HTTPException:
    from backend.tools.common import redact_secrets
    message = redact_secrets(str(exc))
    if "not found" in message.lower():
        status_code = 404
    elif any(term in message.lower() for term in ("investigation failed", "could not produce", "model is unavailable")):
        status_code = 502
    else:
        status_code = 409
    return HTTPException(status_code=status_code, detail=message)


@app.post("/tasks", status_code=201)
def create_agent_task(
    request: TaskCreateRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    background: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        if background:
            task = start_task(request.project_id, request.request, request.mode)
            response.status_code = 202
            background_tasks.add_task(process_task, task["task_id"])
            return task
        return create_task(request.project_id, request.request, request.mode)
    except TaskError as exc:
        raise _task_error(exc) from exc


@app.get("/tasks/{task_id}")
def agent_task_status(task_id: str) -> dict[str, Any]:
    try:
        return get_task(task_id)
    except TaskError as exc:
        raise _task_error(exc) from exc


@app.get("/tasks")
def agent_tasks(project_id: str | None = None, limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    return {"tasks": list_tasks(project_id, limit)}


@app.post("/tasks/{task_id}/approve")
def approve_agent_task(task_id: str) -> dict[str, Any]:
    try:
        return approve_task(task_id)
    except TaskError as exc:
        raise _task_error(exc) from exc


@app.post("/tasks/{task_id}/reject")
def reject_agent_task(task_id: str) -> dict[str, Any]:
    try:
        return reject_task(task_id)
    except TaskError as exc:
        raise _task_error(exc) from exc


@app.post("/tasks/{task_id}/rollback")
def rollback_agent_task(task_id: str) -> dict[str, Any]:
    try:
        return rollback_task(task_id)
    except TaskError as exc:
        raise _task_error(exc) from exc


def _index_project_background(project_id: str) -> None:
    try:
        index_registered_project(project_id)
    except Exception:
        logger.exception("Background indexing failed for project %s", project_id)


@app.post("/projects/index")
def index_project(request: ProjectIndexRequest, background_tasks: BackgroundTasks, response: Response, background: bool = Query(default=False)) -> dict[str, Any]:
    try:
        project = register_project(request.project_path)
        if background:
            from backend.project_manager import update_project
            update_project(project["id"], status="indexing", indexing_error=None, indexing_progress={"stage": "Queued", "files_discovered": 0, "files_processed": 0, "chunks_created": 0, "embedding_batches_completed": 0, "total_embedding_batches": 0, "vectors_stored": 0})
            background_tasks.add_task(_index_project_background, project["id"])
            response.status_code = 202
            return get_project(project["id"])
        return index_registered_project(project["id"])
    except InvalidProjectPathError as exc:
        raise HTTPException(status_code=400, detail={"code": "PROJECT_PATH_INVALID", "message": str(exc)}) from exc
    except ProjectAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EmptyProjectError as exc:
        raise HTTPException(status_code=400, detail={"code": "INDEXING_FAILED", "message": str(exc)}) from exc
    except EmbeddingUnavailableError as exc:
        logger.warning("Embedding service unavailable: %s", type(exc).__name__)
        message = str(exc)
        code = "OLLAMA_TIMEOUT" if "timed out" in message.lower() or "timeout" in message.lower() else "OLLAMA_UNAVAILABLE"
        raise HTTPException(status_code=503, detail={"code": code, "message": message}) from exc
    except EmbeddingError as exc:
        logger.warning("Embedding generation failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail={"code": "EMBEDDING_FAILED", "message": str(exc)}) from exc
    except ChromaIndexError as exc:
        logger.exception("ChromaDB indexing failure")
        raise HTTPException(status_code=500, detail={"code": "CHROMA_FAILED", "message": str(exc)}) from exc
    except (requests.RequestException, RuntimeError) as exc:
        logger.warning("Indexing service failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail={"code": "INDEXING_FAILED", "message": "Indexing service is unavailable. Check Ollama and retry indexing."}) from exc
    except Exception as exc:
        logger.exception("Unexpected project indexing failure")
        raise HTTPException(status_code=500, detail={"code": "INDEXING_FAILED", "message": "Indexing could not complete. Check the backend logs and retry."}) from exc


@app.get("/projects")
def projects() -> dict[str, list[dict[str, Any]]]:
    return {"projects": list_projects()}


@app.get("/projects/{project_id}")
def project_detail(project_id: str) -> dict[str, Any]:
    project = get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' was not found")
    return project


@app.delete("/projects/{project_id}")
def remove_project(project_id: str) -> dict[str, Any]:
    try:
        removed = delete_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project was not found") from exc
    except Exception as exc:
        logger.exception("Project removal failed for %s", project_id)
        raise HTTPException(status_code=500, detail="Project could not be removed; its source files were not deleted.") from exc
    return {
        "removed": True,
        "project_id": project_id,
        "name": removed.get("name", project_id),
        "source_files_preserved": True,
    }


@app.get("/projects/{project_id}/git")
def project_git(project_id: str) -> dict[str, Any]:
    if get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' was not found")
    try:
        return inspect_git(project_id)
    except GitInspectionError as exc:
        logger.warning("Git inspection failed for project %s", project_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tools")
def available_tools() -> dict[str, list[dict[str, Any]]]:
    return {"tools": list_tools()}


@app.post("/tools/{tool_name}")
def call_tool(tool_name: str, payload: dict[str, Any]) -> Any:
    tool = get_tool(tool_name)
    schema = get_tool_schema(tool_name)
    if tool is None or schema is None:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' was not found")
    try:
        arguments = schema.model_validate(payload)
    except ValidationError as exc:
        errors = [
            {"location": error["loc"], "message": error["msg"]}
            for error in exc.errors()
        ]
        raise HTTPException(status_code=400, detail={"message": "Invalid tool input", "errors": errors}) from exc

    project_id = getattr(arguments, "project_id", None)
    if project_id and get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' was not found")

    result = tool(**arguments.model_dump())
    if result["success"]:
        return result

    logger.warning("Registered tool %s failed", tool_name)
    error = result["error"] or "Tool execution failed"
    lowered_error = error.lower()
    if "not found" in lowered_error:
        status_code = 404
    elif "unavailable" in lowered_error or "not installed" in lowered_error:
        status_code = 503
    elif "failed" in lowered_error or "unable to" in lowered_error:
        status_code = 500
    else:
        status_code = 400
    raise HTTPException(status_code=status_code, detail=error)

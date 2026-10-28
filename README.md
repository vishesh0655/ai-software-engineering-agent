# Buildmate — AI Software Engineering Agent

Buildmate is a local-first codebase assistant. It indexes registered repositories, answers questions with source references, investigates engineering requests through a bounded tool-using agent, and prepares human-approved code changes. Changes are diffed, backed up, tested, and can be rolled back.

## Features

- React workspace for projects, codebase questions, agent tasks, diffs, approval, test results, rollback, and task history.
- FastAPI backend with project-isolated ChromaDB collections and Ollama embeddings / generation.
- Registered tools for code search, file reading, symbol lookup, project inspection, and test execution.
- Investigation, debugging, review, and proposal workflows. Generated proposals cannot write files directly.
- Approval-gated application with source hash checks, atomic writes, backups, test execution, bounded repair attempts, and rollback.
- Read-only Git branch/status/diff inspection. No push or commit capability.
- Environment-backed model, storage, API, and agent limits.

## Architecture

```text
React + Vite -> FastAPI
                   |-- RAG -> Ollama embeddings -> ChromaDB
                   |-- Agent Core -> fixed registered tool set
                   |-- Task JSON -> approval/apply/test/rollback
                   `-- Read-only Git inspection (fixed arguments)
```

The backend uses Python 3.13, FastAPI, Pydantic, Requests, ChromaDB, and Ollama. Embeddings are sent in configurable batches (16 by default) and retried on transient connection, timeout, and service failures. The frontend uses React, Vite, and Lucide icons. Task and project metadata are JSON files under `data/`. Chroma vectors and backups are stored under `data/`.

## Local installation

### Python environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.11+ is expected; Python 3.13 is the verified environment.

### Ollama

Install [Ollama](https://ollama.com/), run it, and pull the configured models:

```powershell
ollama pull gemma4:latest
ollama pull nomic-embed-text
```

Embeddings are used during indexing/retrieval. The generation model powers answers and agent decisions. Change model names in `.env` if using different locally installed models.

### Configuration

The application has working defaults and does not require a `.env` file. To override them, create a `.env` file in the repository root and add only the settings you want to change:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama service base URL |
| `OLLAMA_BASE_URL_DOCKER` | `http://host.docker.internal:11434` | Ollama URL used by Docker Compose |
| `OLLAMA_GENERATION_MODEL` | `gemma4:latest` | Text generation model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDING_BATCH_SIZE` | `16` | Texts per embedding request |
| `EMBEDDING_TIMEOUT` | `120` | Embedding request timeout in seconds |
| `EMBEDDING_MAX_RETRIES` | `3` | Retries after transient embedding errors |
| `CHROMA_PATH` | `data/chroma` | Persistent vector database directory |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8001` | API bind address and port |
| `AGENT_MAX_ITERATIONS` | `8` | Maximum model action iterations |
| `AGENT_MAX_TOOL_CALLS` | `8` | Maximum registered tool calls |
| `AGENT_MAX_INVALID_ACTIONS` | `2` | Invalid actions before stop |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8001` | Browser-visible API URL (build time) |

`.env` is ignored by Git. Keep any credentials or private settings there, not in source files.

### Start backend and frontend

From the repository root, start the API:

```powershell
python -m backend.main
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. API docs are at `http://127.0.0.1:8001/docs`. For development reload, use `python -m uvicorn backend.api:app --host 127.0.0.1 --port 8001 --reload`.

### Docker

Docker Compose runs the API and built React UI. Ollama remains on the host. Start with the default settings:

```powershell
docker compose up --build
```

The UI is at `http://localhost:5173`; the API is at `http://localhost:8000` by default. Docker uses `OLLAMA_BASE_URL_DOCKER` to reach Ollama. Mount project repositories below `./projects` (or set `PROJECTS_ROOT` to a host directory), then register their **container paths** under `/workspace/<repository>`. If you change the API port or address, set `VITE_API_BASE_URL` to an address accessible by the browser.

## Example workflow

1. Register a project by entering its local path in the sidebar. Indexing reads supported source files and creates a project-specific vector collection.
2. Ask a codebase question for an answer and indexed source locations.
3. Start an agent task in automatic, investigate, code review, or propose-fix mode.
4. Review tool calls, diagnosis, affected files, reasons, and the generated unified diff.
5. Approve to apply the proposal and run the registered test tool, or reject without file changes.
6. Roll back an applied change from its recorded backup, while files still match the applied content.
7. Inspect Git status and diff in the read-only panel.

## API overview

- `GET /health`, `GET /tools`
- `GET /projects`, `GET /projects/{project_id}`, `POST /projects/index`
- `DELETE /projects/{project_id}` (unregisters metadata and deletes only that project's Chroma collection; source files stay in place)
- `POST /ask` (legacy global collection without `project_id`; project collection otherwise)
- `POST /agent` (bounded tool investigation)
- `POST /tasks`, `GET /tasks`, `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/approve`, `/reject`, `/rollback`
- `GET /projects/{project_id}/git` (read-only branch, changed files, and diff)
- `POST /tools/{tool_name}`

Task status is available through polling `GET /tasks/{task_id}`. The UI creates tasks with `POST /tasks?background=true`, which returns `202` and a task ID immediately; progress and tool calls are persisted as the bounded agent runs. Existing clients may omit the query parameter to retain the synchronous `201` response. During approval, status records `testing` and `analyzing_test_failure`; the UI polls while approval is running.

Project indexing accepts `POST /projects/index?background=true` and returns the project immediately. `GET /projects/{project_id}` includes persisted `indexing_progress` fields for the current stage, files discovered/processed, chunks created, embedding batches completed, and vectors stored. The UI polls this state and shows actionable errors. Indexing writes vectors only after all embedding batches have been validated, then removes stale IDs after upserting current IDs. Stable chunk IDs make retries idempotent.

### Indexing troubleshooting

If indexing fails:

1. Check Ollama with `ollama list`.
2. Check the service at `http://127.0.0.1:11434/api/tags`.
3. Confirm `nomic-embed-text` is installed (`ollama pull nomic-embed-text`).
4. Restart Ollama if necessary.
5. Retry indexing; transient embedding failures are retried automatically.

## Safety model

- The agent calls only registered tools. Shell and Python execution are not exposed to the model.
- The proposal model returns structured relative paths and exact source excerpts. The server resolves paths under the selected registered project, checks file type/size and source uniqueness, and generates the diff itself.
- No source changes until explicit human approval. Apply rechecks original file hashes, makes backups, and writes atomically. Test results come from the registered test tool; failed tests are recorded as failures, and follow-up proposals require fresh approval.
- Removing a project only removes its registered metadata and validated project-specific Chroma collection. It never removes the registered source directory and leaves other project collections and metadata alone.
- Rollback verifies the current applied hash and restores verified backups.
- Git inspection invokes fixed read-only commands with `shell=False`; it has no commit or push operations.
- Tool file access checks project containment and filters sensitive paths. Secret-like strings are redacted in exposed diffs and test output.
- Ollama and indexing errors are mapped to service errors. Python logging is used without returning stack traces to callers.

## Project structure

```text
backend/          FastAPI, RAG/indexing, agent, task service, and tools
frontend/         React/Vite interface
tests/            pytest backend and API workflow tests
data/projects.json
data/tasks.json   Local JSON stores; backups and Chroma data are generated
Dockerfile        Python API image
docker-compose.yml API + UI deployment
```

## Test and build

```powershell
python -m pytest -q
cd frontend
npm run build
```

## Limitations and future work

Task/project JSON persistence is intended for one local instance, not concurrent multi-worker production deployments. There is no authentication; bind the API to a trusted network. Initial task investigation is synchronous, while status is persisted and polled during subsequent actions. Git status includes working-tree paths and diff; it never stages, commits, or pushes. Ollama runs separately. Production deployment should add authentication, multi-user storage, background job workers, secrets management, and deployment-specific browser/API origin policy.

"""Environment-backed runtime settings."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    llm_model: str = os.getenv("OLLAMA_GENERATION_MODEL", os.getenv("OLLAMA_MODEL", "gemma4:latest"))
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    embedding_batch_size: int = _integer("EMBEDDING_BATCH_SIZE", 16, 1, 128)
    embedding_timeout: int = _integer("EMBEDDING_TIMEOUT", 120, 1, 600)
    embedding_max_retries: int = _integer("EMBEDDING_MAX_RETRIES", 3, 0, 10)
    chroma_path: str = str((ROOT / os.getenv("CHROMA_PATH", "data/chroma")).resolve())
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = _integer("API_PORT", 8001, 1, 65535)
    agent_max_iterations: int = _integer("AGENT_MAX_ITERATIONS", 8, 1, 32)
    agent_max_tool_calls: int = _integer("AGENT_MAX_TOOL_CALLS", 8, 1, 32)
    agent_max_invalid_actions: int = _integer("AGENT_MAX_INVALID_ACTIONS", 2, 1, 8)


settings = Settings()

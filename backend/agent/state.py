from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    project_id: str
    user_request: str
    iteration_count: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None

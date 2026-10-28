import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.tools.registry import get_tool, get_tool_schema


class ProtocolError(ValueError):
    """The model did not return a valid, permitted agent action."""

    def __init__(
        self,
        message: str,
        *,
        response_length: int | None = None,
        response_start_kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.response_length = response_length
        self.response_start_kind = response_start_kind


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1)
    # Empty arguments are acceptable for registry tools that need no model inputs.
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["final"]
    answer: str = Field(min_length=1)


def _validation_summary(exc: ValidationError) -> str:
    summaries = []
    for error in exc.errors(include_input=False)[:3]:
        location = ".".join(
            "".join(character for character in str(part) if character.isalnum() or character == "_")[:32]
            for part in error.get("loc", ())
        ) or "response"
        if error.get("type") == "missing":
            summary = f"{location} is required"
        elif error.get("type") == "extra_forbidden":
            summary = f"{location} is not an allowed field"
        else:
            summary = f"{location} is invalid"
        summaries.append(summary)
    return "; ".join(summaries) or "response fields are invalid"


def _strip_markdown_fence(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        inner = stripped.strip("`")
        if "\n" in inner:
            inner = inner.split("\n", 1)[1].rsplit("\n", 1)[0] if inner.count("\n") >= 2 else inner
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        return inner.strip()
    return stripped


def _looks_like_observation(decoded: dict[str, Any]) -> bool:
    if not isinstance(decoded, dict):
        return False
    observation_keys = {"tool", "success", "result", "error", "data", "status", "summary", "details"}
    if "action" in decoded:
        return False
    return bool(set(decoded) & observation_keys)


def validate_action(raw: str, project_id: str) -> FinalAction | tuple[str, dict[str, Any]]:
    if not isinstance(raw, str):
        raise ProtocolError("The model response was not text")
    if not raw.strip():
        raise ProtocolError(
            "The model response was empty or whitespace",
            response_length=len(raw),
            response_start_kind="empty",
        )
    cleaned = _strip_markdown_fence(raw)
    try:
        decoded = json.loads(cleaned)
    except (TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            detail = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
            prefix = cleaned.lstrip()[:1]
            start_kind = (
                "markdown" if prefix == "`" else
                "object" if prefix == "{" else
                "array" if prefix == "[" else
                "text" if prefix.isalpha() else
                "markup" if prefix == "<" else
                "non_json_symbol"
            )
            raise ProtocolError(
                f"The model response was not valid JSON: {detail}",
                response_length=len(raw), response_start_kind=start_kind,
            ) from exc
        raise ProtocolError("The model response was not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ProtocolError("The model response must be a JSON object")
    if _looks_like_observation(decoded):
        raise ProtocolError(
            "The model response looked like an observation object, not an action envelope. "
            "Return exactly {\"action\":\"<tool_name>\",\"arguments\":{...}} or {\"action\":\"final\",\"answer\":\"...\"}."
        )

    if decoded.get("action") == "final":
        try:
            final = FinalAction.model_validate(decoded)
        except ValidationError as exc:
            raise ProtocolError("Malformed final response") from exc
        answer = final.answer.strip()
        if not answer:
            raise ProtocolError("Final answer must not be empty")
        return FinalAction(action="final", answer=answer)

    try:
        action = ToolAction.model_validate(decoded)
    except ValidationError as exc:
        raise ProtocolError(f"Malformed tool action: {_validation_summary(exc)}") from exc
    if action.action == "final" or get_tool(action.action) is None:
        raise ProtocolError(f"Unknown action: {action.action}")
    schema = get_tool_schema(action.action)
    if schema is None:
        raise ProtocolError(f"Unknown action: {action.action}")
    if "project_id" in action.arguments:
        raise ProtocolError("Tool arguments must not select or override project_id")
    try:
        validated = schema.model_validate({**action.arguments, "project_id": project_id})
    except ValidationError as exc:
        raise ProtocolError(f"Tool arguments failed validation: {_validation_summary(exc)}") from exc
    return action.action, validated.model_dump()

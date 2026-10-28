import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from backend.agent.prompts import build_prompt
from backend.agent.protocol import FinalAction, ProtocolError, validate_action
from backend.agent.state import AgentState
from backend.llm import ModelGenerationError, generate
from backend.tools.registry import get_tool
from backend.settings import settings


@dataclass(frozen=True)
class AgentLimits:
    max_iterations: int = settings.agent_max_iterations
    max_tool_calls: int = settings.agent_max_tool_calls
    max_invalid_actions: int = settings.agent_max_invalid_actions
    max_observations: int = 6
    max_observation_chars: int = 4000
    max_total_observation_chars: int = 12000


DEFAULT_LIMITS = AgentLimits()
logger = logging.getLogger("backend.agent.agent")


class AgentError(RuntimeError):
    """The agent could not complete within its configured safety limits."""


def _tool_signature(call: dict[str, Any]) -> tuple[str, str]:
    arguments = call.get("arguments", {})
    return call.get("tool", ""), json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)


def _repeated_tool_loop(state: AgentState) -> bool:
    if len(state.tool_calls) < 2:
        return False
    recent = state.tool_calls[-4:]
    signatures = [_tool_signature(call) for call in recent]
    if len(set(signatures)) <= 1 and len(signatures) >= 2:
        return True
    if len(recent) >= 2 and recent[-1].get("tool") == recent[-2].get("tool"):
        previous = json.dumps(recent[-2].get("arguments", {}), ensure_ascii=False, sort_keys=True, default=str)
        current = json.dumps(recent[-1].get("arguments", {}), ensure_ascii=False, sort_keys=True, default=str)
        if previous == current:
            return True
    return False


def _maybe_synthesize_final_answer(state: AgentState, user_request: str, project_id: str) -> dict[str, Any]:
    if not state.tool_calls:
        raise AgentError("Agent investigation produced no usable evidence")
    evidence = [{
        "tool": call.get("tool"),
        "arguments": call.get("arguments"),
        "success": call.get("success"),
        "result": call.get("result"),
    } for call in state.tool_calls]
    prompt = (
        "You have enough evidence to answer the user's request. Stop calling tools and answer from the evidence only. "
        "If the evidence is insufficient or contradictory, state that clearly. Return exactly one JSON object with the "
        "shape {\"action\":\"final\",\"answer\":\"...\"}. No markdown or extra text.\n\n"
        f"PROJECT_ID:\n{project_id}\n\nUSER_REQUEST:\n{user_request}\n\nTOOL_EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )
    raw_action = generate(prompt, response_format="json")
    validated = validate_action(raw_action, project_id)
    if not isinstance(validated, FinalAction):
        raise AgentError("Agent final synthesis returned a non-final action")
    state.final_answer = validated.answer
    return {
        "project_id": state.project_id,
        "answer": state.final_answer,
        "tool_calls": state.tool_calls,
        "iterations": state.iteration_count,
    }


def _bounded_observations(state: AgentState, limits: AgentLimits) -> list[dict[str, Any]]:
    recent = state.observations[-limits.max_observations:]
    bounded = []
    remaining = limits.max_total_observation_chars
    for observation in reversed(recent):
        encoded = json.dumps(observation, ensure_ascii=False, default=str)
        if remaining <= 0:
            break
        value = encoded[: min(limits.max_observation_chars, remaining)]
        bounded.append({"observation": value})
        remaining -= len(value)
    return list(reversed(bounded))


def run_agent(
    project_id: str,
    question: str,
    *,
    limits: AgentLimits = DEFAULT_LIMITS,
    on_tool_call: Callable[[dict[str, Any]], None] | None = None,
    on_iteration: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    request = question.strip()
    if not request:
        raise AgentError("Question must not be empty")
    state = AgentState(project_id=project_id, user_request=request)
    invalid_actions = 0
    feedback: str | None = None

    while state.iteration_count < limits.max_iterations:
        state.iteration_count += 1
        if on_iteration is not None:
            on_iteration(state.iteration_count)
        prompt = build_prompt(
            state.project_id, state.user_request,
            _bounded_observations(state, limits), feedback,
        )
        feedback = None
        try:
            raw_action = generate(prompt, response_format="json")
            validated = validate_action(raw_action, state.project_id)
        except ProtocolError as exc:
            invalid_actions += 1
            feedback = str(exc)
            if invalid_actions >= limits.max_invalid_actions:
                if state.tool_calls:
                    try:
                        return _maybe_synthesize_final_answer(state, state.user_request, state.project_id)
                    except AgentError:
                        pass
                raise AgentError(f"Agent stopped after repeated invalid actions: {exc}") from exc
            continue
        except ModelGenerationError as exc:
            if state.tool_calls:
                try:
                    return _maybe_synthesize_final_answer(state, state.user_request, state.project_id)
                except AgentError:
                    pass
            raise AgentError(f"Agent model returned no usable response: {exc}") from exc
        except RuntimeError as exc:
            if state.tool_calls:
                try:
                    return _maybe_synthesize_final_answer(state, state.user_request, state.project_id)
                except AgentError:
                    pass
            raise AgentError(f"Agent model is unavailable: {exc}") from exc

        if isinstance(validated, FinalAction):
            if not state.tool_calls:
                invalid_actions += 1
                feedback = "At least one registered tool must run before the final answer"
                if invalid_actions >= limits.max_invalid_actions:
                    raise AgentError("Agent stopped after repeated invalid actions: final answer lacked tool evidence")
                continue
            state.final_answer = validated.answer
            return {
                "project_id": state.project_id,
                "answer": state.final_answer,
                "tool_calls": state.tool_calls,
                "iterations": state.iteration_count,
            }

        invalid_actions = 0
        if len(state.tool_calls) >= limits.max_tool_calls:
            raise AgentError("Agent reached the maximum number of tool calls")
        tool_name, arguments = validated
        tool = get_tool(tool_name)
        if tool is None:  # Defensive: protocol validation checks the same registry.
            raise AgentError("Validated tool is no longer registered")
        started = time.monotonic()
        try:
            result = tool(**arguments)
        except Exception as exc:
            logger.warning(
                "Agent tool raised an exception project_id=%s iteration=%s tool=%s error_type=%s",
                project_id, state.iteration_count, tool_name, type(exc).__name__,
            )
            result = {"success": False, "tool": tool_name, "data": None,
                      "error": f"Tool execution failed: {type(exc).__name__}"}
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        observation = serialized[:limits.max_observation_chars]
        call = {
            "iteration": state.iteration_count,
            "tool": tool_name,
            "arguments": arguments,
            "success": bool(result.get("success")),
            "elapsed_ms": elapsed_ms,
            "result": observation,
        }
        state.tool_calls.append(call)
        if on_tool_call is not None:
            on_tool_call(call)
        logger.info(
            "Agent tool completed project_id=%s iteration=%s tool=%s success=%s elapsed_ms=%s",
            project_id, state.iteration_count, tool_name, call["success"], elapsed_ms,
        )
        state.observations.append({
            "tool": tool_name,
            "arguments": arguments,
            "success": bool(result.get("success")),
            "result": observation,
        })

        if _repeated_tool_loop(state):
            logger.info(
                "Agent detected repetitive tool loop project_id=%s iteration=%s tool=%s evidence_count=%s",
                project_id, state.iteration_count, tool_name, len(state.tool_calls),
            )
            try:
                return _maybe_synthesize_final_answer(state, state.user_request, state.project_id)
            except AgentError:
                pass

    if state.tool_calls:
        try:
            return _maybe_synthesize_final_answer(state, state.user_request, state.project_id)
        except AgentError:
            pass

    raise AgentError("Agent reached the maximum number of iterations without a final answer")

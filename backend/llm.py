import logging
import requests
from backend.settings import settings


OLLAMA_URL = f"{settings.ollama_url}/api/generate"
MODEL = settings.llm_model
logger = logging.getLogger("backend.llm")


class ModelGenerationError(RuntimeError):
    """Ollama returned a successful response without usable generated text."""


def _normalize_ollama_format(response_format):
    if response_format is None:
        return None
    if isinstance(response_format, str):
        return response_format.lower() if response_format.lower() in {"json", "text"} else response_format
    return response_format


def _safe_ollama_metadata(response, payload, body=None):
    metadata = {
        "model": MODEL,
        "status": getattr(response, "status_code", None),
        "stream": bool(payload.get("stream")),
        "format": payload.get("format"),
        "response_type": type(body.get("response") if isinstance(body, dict) else None).__name__ if isinstance(body, dict) else None,
        "response_length": len(body.get("response", "")) if isinstance(body, dict) and isinstance(body.get("response"), str) else None,
        "response_whitespace": bool(isinstance(body, dict) and isinstance(body.get("response"), str) and not body.get("response").strip()),
        "keys": list(body.keys()) if isinstance(body, dict) else None,
        "done": body.get("done") if isinstance(body, dict) else None,
        "done_reason": body.get("done_reason") if isinstance(body, dict) else None,
        "total_duration": body.get("total_duration") if isinstance(body, dict) else None,
        "error": body.get("error") if isinstance(body, dict) else None,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def generate(prompt, response_format=None, *, stream=False, timeout=120, max_retries=1):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": bool(stream),
    }
    normalized = _normalize_ollama_format(response_format)
    if normalized is not None:
        payload["format"] = normalized

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                continue
            raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {exc}") from exc

        if not response.ok:
            body = response.text.strip() or "<empty response body>"
            logger.warning(
                "Ollama request failed model=%s status=%s stream=%s format=%s error=%s",
                MODEL, getattr(response, "status_code", None), bool(payload.get("stream")), payload.get("format"), body[:200],
            )
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {body}")

        try:
            body = response.json()
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "Ollama returned malformed JSON model=%s status=%s response_text_preview=%s",
                MODEL, getattr(response, "status_code", None), (response.text[:200] if hasattr(response, "text") else "")
            )
            raise RuntimeError(f"Ollama returned an invalid generate response: {response.text}") from exc

        logger.info(
            "Ollama generation diagnostics model=%s status=%s keys=%s response_length=%s whitespace=%s done=%s stream=%s format=%s",
            MODEL,
            getattr(response, "status_code", None),
            list(body.keys()) if isinstance(body, dict) else None,
            len(body.get("response", "")) if isinstance(body, dict) and isinstance(body.get("response"), str) else None,
            bool(isinstance(body, dict) and isinstance(body.get("response"), str) and not body.get("response").strip()),
            body.get("done") if isinstance(body, dict) else None,
            bool(payload.get("stream")),
            payload.get("format"),
        )

        if not isinstance(body, dict):
            if attempt < max_retries:
                continue
            raise ModelGenerationError("Ollama returned a non-dictionary JSON payload.")

        generated = body.get("response")
        if not isinstance(generated, str):
            if attempt < max_retries:
                continue
            raise ModelGenerationError("Ollama returned no text field in its generate response.")
        if not generated.strip():
            metadata = {
                key: body.get(key)
                for key in ("done", "done_reason", "prompt_eval_count", "eval_count", "total_duration")
                if body.get(key) is not None
            }
            logger.warning(
                "Ollama returned an empty model response model=%s metadata=%s response_preview=%r",
                MODEL, metadata, generated[:80],
            )
            if attempt < max_retries:
                continue
            details = ", ".join(f"{key}={value}" for key, value in metadata.items())
            suffix = f" ({details})" if details else ""
            raise ModelGenerationError(f"Ollama returned an empty model response{suffix}.")
        return generated

    if last_error is not None:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {last_error}")
    raise ModelGenerationError("Ollama returned no usable text after retries.")

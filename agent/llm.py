"""
Ollama wrapper for qwen3:8b — synchronous, minimal.
"""
import re
import logging
import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
DEFAULT_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 500,
}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 emits by default.

    If the only content is inside <think>, extract any JSON from within it
    as the model may have put its answer there.
    """
    if "</think>" in text:
        after = text.split("</think>", 1)[1].strip()
        if after:
            return after
        # Everything was inside <think> — try to salvage JSON from it
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            inner = think_match.group(1).strip()
            # Look for JSON inside the thinking block
            json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', inner)
            if json_match:
                return json_match.group()
            json_match = re.search(r'\{[^{}]*"done"[^{}]*\}', inner, re.DOTALL)
            if json_match:
                return json_match.group()
    return text.strip()


def chat(messages: list[dict], timeout: float = 60.0) -> str:
    """Send messages to qwen3:8b via Ollama. Returns assistant content string.

    Raises RuntimeError if Ollama is unreachable or returns an error.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": DEFAULT_OPTIONS,
        "think": False,  # Disable thinking mode for structured output
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError("Ollama is not running at localhost:11434")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama returned {e.response.status_code}: {e.response.text[:200]}")
    except httpx.TimeoutException:
        raise RuntimeError(f"Ollama timed out after {timeout}s")

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    raw_content = content
    content = _strip_thinking(content)
    if not content:
        logger.warning("LLM returned empty after stripping think tags. Raw: %s", raw_content[:500])
    else:
        logger.info("LLM response (%d chars): %s", len(content), content[:200])
    return content


def health_check() -> bool:
    """Return True if Ollama is running and qwen3:8b is available."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            return any("qwen3" in m for m in models)
    except Exception:
        return False

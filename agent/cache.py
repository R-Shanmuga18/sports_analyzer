"""Development-time Groq response cache for reducing repeated API costs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from groq import Groq


def _cache_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    path = base_dir / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _to_object(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_object(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_object(v) for v in value]
    return value


def get_cache_key(model: str, messages: list, tools: list = None) -> str:
    """Generate a deterministic hash key for this exact LLM call."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools or [],
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cached_response(cache_key: str) -> dict | None:
    """Return cached response dict if it exists, else None."""
    cache_file = _cache_dir() / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached_response(cache_key: str, response: dict) -> None:
    """Save LLM response to cache file."""
    cache_file = _cache_dir() / f"{cache_key}.json"
    model_name = response.get("model", "") if isinstance(response, dict) else ""
    messages_preview = ""
    if isinstance(response, dict):
        req_messages = response.get("_request_messages", [])
        if req_messages:
            messages_preview = json.dumps(req_messages, ensure_ascii=True)[:100]

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": cache_key,
        "model": model_name,
        "messages_preview": messages_preview,
        "response": response,
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def cached_llm_call(model: str, messages: list, tools: list = None, **kwargs) -> object:
    """
    Make a Groq chat completion call with caching.
    If cache hit: return cached response (reconstructed as object).
    If cache miss: make real API call, cache it, return it.
    Always prints: [CACHE HIT] or [API CALL] so developer can track costs.
    """
    load_dotenv()
    cache_enabled = os.getenv("CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    model_name = (model or "").strip() or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    cache_key = get_cache_key(model=model_name, messages=messages, tools=tools)

    if cache_enabled:
        cached = get_cached_response(cache_key)
        if cached and isinstance(cached, dict) and "response" in cached:
            print(f"[CACHE HIT] {cache_key}")
            return _to_object(cached["response"])

    print(f"[API CALL] {cache_key}")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    client = Groq(api_key=api_key)

    request_payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if tools:
        request_payload["tools"] = tools
    request_payload.update(kwargs)

    response_obj = client.chat.completions.create(**request_payload)
    response_dict = _to_jsonable(response_obj)
    if isinstance(response_dict, dict):
        response_dict["_request_messages"] = messages

    if cache_enabled and isinstance(response_dict, dict):
        save_cached_response(cache_key, response_dict)

    return _to_object(response_dict)

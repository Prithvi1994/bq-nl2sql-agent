"""Model backends for the agentic chat.

`create_llm()` reuses the repo's existing OpenAI-compatible client and adds the one
thing the experiment agent needs that the SQL agent does not: a session header, so the
provider can route the conversation.

No key is hardcoded. The key comes from the environment; a missing key raises with the
name of the variable to set rather than silently falling back to a mock.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Protocol, Sequence

Messages = List[Dict[str, str]]


class LLM(Protocol):
    name: str

    def complete(self, messages: Messages) -> str: ...


class LLMError(RuntimeError):
    pass


class ChatLLM:
    """OpenAI-compatible chat endpoint with an optional extra header."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
        extra_headers: Optional[Dict[str, str]] = None,
        session: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise LLMError(
                f"no API key for {model}. Set it in the environment and pass api_key_env."
            )
        self.model = model
        self.name = f"chat:{model}"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.session = session or f"abtest-{uuid.uuid4().hex[:12]}"
        self.extra_headers = dict(extra_headers or {})
        self.calls: List[Messages] = []
        self.last_usage: Dict[str, Any] = {}

    def complete(self, messages: Messages) -> str:
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-opencode-session": self.session,
        }
        headers.update(self.extra_headers)
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        t0 = time.perf_counter()
        r = requests.post(
            f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout_s
        )
        if r.status_code != 200:
            raise LLMError(f"{r.status_code} from {self.base_url}: {r.text[:300]}")
        data = r.json()
        self.last_usage = {
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": data.get("usage", {}).get("completion_tokens"),
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in response: {str(data)[:200]}")
        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        # Some gateways return reasoning separately and leave content empty.
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        self.calls.append(messages)
        return content


class ScriptedLLM:
    """Replays a fixed list of replies. For harness tests only, never for eval scores."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.name = "scripted"
        self.calls: List[Messages] = []
        self.last_usage: Dict[str, Any] = {}

    def complete(self, messages: Messages) -> str:
        self.calls.append(messages)
        if not self._replies:
            return ""
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


def create_llm(
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: str = "OPENCODE_GO_API_KEY",
    **kwargs: Any,
) -> LLM:
    """Build a chat backend from the environment.

    Defaults to the provider this repo already uses. An API key is required: a
    missing key is an error, not a reason to fall back to a mock, because a mock
    silently turns a broken deployment into a passing test.
    """
    if model == "scripted":
        return ScriptedLLM(kwargs.get("replies", []))

    model = model or os.environ.get("NL2SQL_CHAT_MODEL", "space-bunny-free")
    base_url = base_url or os.environ.get("NL2SQL_CHAT_BASE_URL", "https://opencode.ai/zen/go/v1")
    key = api_key or os.environ.get(api_key_env, "")
    if not key and api_key_env != "OPENROUTER_API_KEY":
        raise LLMError(
            f"{api_key_env} is not set. Export it (it is in /opt/data/.env) or pass api_key=."
        )
    return ChatLLM(model=model, base_url=base_url, api_key=key, **kwargs)

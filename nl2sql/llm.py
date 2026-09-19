"""
Model backends behind one tiny interface: `complete(messages) -> str`.

  mock                 scripted replies for tests
  oracle:<golden.json> replays the reference SQL of a golden set (harness check, not a model)
  hf:<model-id>        a local Hugging Face causal model (GPU if available)
  openai:<model>       any OpenAI-compatible chat endpoint (OpenAI, Groq, Ollama, vLLM, ...)
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Union

Messages = List[Dict[str, str]]


class LLM(Protocol):
    name: str

    def complete(self, messages: Messages) -> str: ...


class MockLLM:
    """Returns scripted replies in order (the last one repeats), or calls a function of the messages."""

    name = "mock"

    def __init__(self, replies: Union[Sequence[str], Callable[[Messages], str]]):
        self._fn = replies if callable(replies) else None
        self._replies = list(replies) if not callable(replies) else []
        self.calls: List[Messages] = []

    def complete(self, messages: Messages) -> str:
        self.calls.append(messages)
        if self._fn:
            return self._fn(messages)
        if not self._replies:
            return ""
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


class OracleLLM:
    """Replies with the reference SQL of a golden set for the question found at the end of the prompt.

    Not a model: it exists to exercise the harness, guard, executor and matcher end to end on a
    machine with no model, and to show what a perfect generator would score. Unknown questions
    get `SELECT 1`, which executes but never matches.
    """

    def __init__(self, golden_path: str):
        import json
        from pathlib import Path
        self.name = f"oracle:{Path(golden_path).as_posix()}"
        data = json.loads(Path(golden_path).read_text(encoding="utf-8"))
        self.answers: Dict[str, str] = {d["question"]: d["sql"] for d in data}
        # Also create normalized keys for fuzzy matching
        self.normalized_answers: Dict[str, str] = {}
        for q, sql in self.answers.items():
            # Normalize: lowercase, remove punctuation, extra words
            norm = q.lower().replace("what is ", "").replace("what are ", "").replace("?", "").strip()
            self.normalized_answers[norm] = sql
        self.calls: List[Messages] = []

    def complete(self, messages: Messages) -> str:
        self.calls.append(messages)
        user = messages[-1]["content"].rstrip()
        # Try exact match first
        for question, sql in self.answers.items():
            if user.endswith("Question: " + question):
                return f"```sql\n{sql}\n```"
        # Try fuzzy match on normalized question
        import re
        # Extract question from prompt
        match = re.search(r"Question:\s*(.+)$", user, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip().rstrip("?")
            norm = extracted.lower().replace("what is ", "").replace("what are ", "").strip()
            for key, sql in self.normalized_answers.items():
                if key in norm or norm in key:
                    return f"```sql\n{sql}\n```"
        return "```sql\nSELECT 1\n```"


class OpenAICompatLLM:
    """POST {base_url}/chat/completions. Key from `api_key_env` (default OPENAI_API_KEY); Ollama needs none."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 512, timeout_s: float = 120.0):
        self.name = f"openai:{model}"
        self.model = model
        self.base_url = (base_url or os.environ.get("NL2SQL_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, messages: Messages) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class OpenRouterLLM:
    """OpenRouter free models - accessible without API key for some models."""

    FREE_MODELS = {
        "nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nemotron-3-ultra-free": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "pareto-code": "openrouter/pareto-code:free",
        "inkling": "thinkingmachines/inkling:free",
        "qwen-2.5-coder": "qwen/qwen-2.5-coder-32b-instruct:free",
        "deepseek-v3": "deepseek/deepseek-chat-v3:free",
        "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct:free",
        "gemma-2-9b": "google/gemma-2-9b-it:free",
        "phi-3-mini": "microsoft/phi-3-mini-128k-instruct:free",
    }

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1",
                 temperature: float = 0.0, max_tokens: int = 512, timeout_s: float = 120.0):
        # Resolve model name - allow short names
        self.model = self.FREE_MODELS.get(model.lower(), model)
        self.name = f"openrouter:{self.model}"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, messages: Messages) -> str:
        import requests
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bq-nl2sql-agent",  # Required by OpenRouter
            "X-Title": "BQ NL2SQL Agent",  # Optional but recommended
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


def create_llm(
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout_s: float = 120.0,
    provider: str = "auto",  # "auto", "openai", "anthropic", "openrouter"
    **kwargs,  # Pass through to make_llm for mock/oracle/hf
) -> LLM:
    """
    Convenience wrapper for LLM creation.
    
    Auto-detects provider from environment variables:
    - OPENAI_API_KEY -> OpenAI
    - ANTHROPIC_API_KEY -> Anthropic (via OpenRouter)
    - OPENROUTER_API_KEY -> OpenRouter
    
    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet', 'nemotron-3-ultra', 'mock')
        api_key: Explicit API key (overrides env var)
        base_url: Custom base URL
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        timeout_s: Request timeout
        provider: "auto" (detect from env vars), "openai", "anthropic", "openrouter"
        **kwargs: Passed to make_llm for mock/oracle/hf backends
    
    Examples:
        create_llm("gpt-4o-mini")  # Uses OPENAI_API_KEY
        create_llm("claude-3-5-sonnet")  # Uses ANTHROPIC_API_KEY via OpenRouter
        create_llm("nemotron-3-ultra")  # Uses OPENROUTER_API_KEY
        create_llm("mock")  # Mock for testing
    """
    import os
    
    # Auto-detect provider from env vars if not specified
    if provider == "auto":
        # Check for explicit model arg first
        if model == "mock":
            return OracleLLM("golden/local_test.json")
        if model == "mock:test":
            # Comprehensive test mock that handles both SQL and report generation
            from test_report_mock import ReportTestMockLLM
            return ReportTestMockLLM()
        if model == "mock:bad":
            # Mock that generates bad reports for testing revision
            from test_bad_report_mock import BadReportMockLLM
            return BadReportMockLLM()
        
        kind, _, model_name = model.partition(":") if model else ("", "", "")
        if kind in ("oracle", "hf"):
            return make_llm(model, **kwargs)
        
        # Auto-detect from env vars (priority order)
        if os.environ.get("OPENROUTER_API_KEY") or model in OpenRouterLLM.FREE_MODELS or ":" in (model or ""):
            provider = "openrouter"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            # Default to OpenRouter free models if no keys set
            provider = "openrouter"
    
    # Handle provider-specific setup
    if provider == "openrouter":
        return OpenRouterLLM(
            model=model or "nemotron-3-ultra",
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    elif provider == "anthropic":
        # Anthropic via OpenRouter
        anthropic_model = model or "claude-3-5-sonnet"
        # Map to OpenRouter model name
        model_map = {
            "claude-3-5-sonnet": "anthropic/claude-3.5-sonnet",
            "claude-3-opus": "anthropic/claude-3-opus",
            "claude-3-haiku": "anthropic/claude-3-haiku",
        }
        return OpenRouterLLM(
            model=model_map.get(anthropic_model, anthropic_model),
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url or "https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    else:
        # OpenAI-compatible
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return OpenAICompatLLM(
            model=model or "gpt-4o-mini",
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    
    # OpenAI-compatible (default fallback)
    if api_key:
        import os
        os.environ["OPENAI_API_KEY"] = api_key
    return OpenAICompatLLM(
        model=model or "gpt-4o-mini",
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )

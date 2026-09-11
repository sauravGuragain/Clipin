"""LLM providers.

One interface, several backends. Ollama is the default here because the project
is local-only by choice; the cloud providers exist so that decision stays
reversible, not because they are needed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """The provider cannot be reached or is not configured."""


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    elapsed: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    default_model: str

    def is_available(self) -> tuple[bool, str]: ...

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> LLMResult: ...


def _post_json(url: str, payload: dict, timeout: int, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMUnavailable(f"Could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Request to {url} timed out after {timeout}s.") from exc


class OllamaProvider:
    """Local models through Ollama.

    `keep_alive=0` is deliberate and not a performance oversight: this machine
    has 16 GB shared between CPU and GPU, and leaving a 5 GB model resident
    after discovery finishes would collide with the render stage. See
    STACK.md section 3.
    """

    name = "ollama"
    default_model = "qwen3:8b"

    def __init__(
        self,
        host: str | None = None,
        keep_alive: str = "0",
        num_ctx: int | None = None,
        think: bool | None = None,
    ) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.keep_alive = keep_alive
        # Ollama derives a default context from available VRAM and can pick
        # something as small as 4096. num_ctx covers prompt *and* generation,
        # so an undersized window silently discards the oldest tokens - which
        # here is the transcript. The model then answers fluently about text it
        # never saw. Setting this explicitly removes the guesswork.
        self.num_ctx = num_ctx or int(os.environ.get("LLM_NUM_CTX", "8192"))
        # Reasoning models (qwen3, deepseek-r1) emit a thinking pass before
        # answering. Ollama returns it in a separate field so it never reaches
        # the JSON parser, but it is still generated, still costs seconds, and
        # still counts against num_predict - so a long think can truncate the
        # answer that follows it. For structured extraction the reasoning buys
        # little, so it is off by default. Set LLM_THINK=1 to compare.
        if think is None:
            think = os.environ.get("LLM_THINK", "0") not in ("0", "false", "False", "")
        self.think = think

    def is_available(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                json.loads(response.read().decode())
            return True, ""
        except Exception:
            return False, (
                f"Ollama is not responding at {self.host}. "
                "Start it with `ollama serve`, then pull a model "
                "(e.g. `ollama pull qwen3:8b`)."
            )

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> LLMResult:
        import time

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
            # Only assert this for structured calls; free-form generation (hook
            # writing later on) may genuinely benefit from reasoning.
            payload["think"] = self.think

        started = time.monotonic()
        data = _post_json(f"{self.host}/api/chat", payload, timeout)
        elapsed = time.monotonic() - started

        message = data.get("message") or {}
        text = message.get("content", "")
        if not text:
            thinking = message.get("thinking") or ""
            if thinking:
                raise LLMError(
                    "The model produced reasoning but no answer - it ran out of "
                    "output budget while thinking. Raise LLM_MAX_OUTPUT or set "
                    "LLM_THINK=0."
                )
            raise LLMError("Ollama returned an empty message.")

        return LLMResult(
            text=text,
            model=payload["model"],
            provider=self.name,
            elapsed=elapsed,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )


class AnthropicProvider:
    name = "anthropic"
    default_model = "claude-sonnet-4-5"

    def is_available(self) -> tuple[bool, str]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set in .env."
        return True, ""

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> LLMResult:
        import time

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set.")

        payload = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        started = time.monotonic()
        data = _post_json(
            "https://api.anthropic.com/v1/messages", payload, timeout,
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        elapsed = time.monotonic() - started

        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage", {})
        return LLMResult(
            text=text, model=payload["model"], provider=self.name, elapsed=elapsed,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )


class OpenAIProvider:
    name = "openai"
    default_model = "gpt-4o-mini"

    def is_available(self) -> tuple[bool, str]:
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set in .env."
        return True, ""

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> LLMResult:
        import time

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMUnavailable("OPENAI_API_KEY is not set.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        data = _post_json(
            "https://api.openai.com/v1/chat/completions", payload, timeout,
            {"Authorization": f"Bearer {key}"},
        )
        elapsed = time.monotonic() - started

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResult(
            text=text, model=payload["model"], provider=self.name, elapsed=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class StubProvider:
    """Deterministic provider for tests.

    Returns canned responses, including deliberately malformed ones, because
    malformed output is the normal case with small local models and the
    recovery path deserves more testing than the happy path.
    """

    name = "stub"
    default_model = "stub"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[dict] = []
        self._index = 0

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> LLMResult:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})

        if self.responses:
            text = self.responses[min(self._index, len(self.responses) - 1)]
            self._index += 1
        else:
            text = "[]"

        return LLMResult(text=text, model="stub", provider=self.name, elapsed=0.0)


PROVIDERS: dict[str, type] = {
    OllamaProvider.name: OllamaProvider,
    AnthropicProvider.name: AnthropicProvider,
    OpenAIProvider.name: OpenAIProvider,
    StubProvider.name: StubProvider,
}


def get_provider(name: str | None = None):
    if name and name != "auto":
        cls = PROVIDERS.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown LLM provider '{name}'. Available: {', '.join(PROVIDERS)}"
            )
        return cls()

    # Auto prefers local, matching the project's local-first stance. The stub
    # is never selected automatically — it must be asked for by name, so it can
    # never silently stand in for a real model.
    for cls in (OllamaProvider, AnthropicProvider, OpenAIProvider):
        provider = cls()
        available, _ = provider.is_available()
        if available:
            return provider

    reasons = []
    for cls in (OllamaProvider, AnthropicProvider, OpenAIProvider):
        _, reason = cls().is_available()
        reasons.append(f"{cls.name}: {reason}")
    raise LLMUnavailable("No LLM provider is available.\n  " + "\n  ".join(reasons))

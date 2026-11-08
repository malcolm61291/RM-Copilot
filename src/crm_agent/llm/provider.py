"""Provider-agnostic LLM client. The agent depends on `LLMClient`, never on a vendor SDK.

LLM_PROVIDER=offline  -> no LLM; deterministic fallbacks are used (CI, demos without keys)
LLM_PROVIDER=anthropic -> Claude via the Anthropic SDK (ANTHROPIC_API_KEY)
LLM_PROVIDER=openai    -> OpenAI chat completions (OPENAI_API_KEY)
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Protocol

from crm_agent.config import settings
from crm_agent.observability.tracer import span


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str, *, json_mode: bool = False,
                 max_tokens: int = 700, temperature: float = 0.3) -> str: ...


class AnthropicClient:
    def __init__(self, model: str):
        import anthropic

        self.name, self.model = f"anthropic:{model}", model
        self._client = anthropic.Anthropic(timeout=settings.llm_timeout_s)

    def complete(self, system, user, *, json_mode=False, max_tokens=700, temperature=0.3) -> str:
        if json_mode:
            system += "\nRespond with a single JSON object only. No prose, no code fences."
        with span("llm", self.name, prompt_chars=len(system) + len(user)) as s:
            try:
                resp = self._client.messages.create(
                    model=self.model, system=system, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": user}])
            except Exception as exc:
                raise LLMError(str(exc)) from exc
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            s["attrs"]["tokens"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
            return text


class OpenAIClient:
    def __init__(self, model: str):
        import openai

        self.name, self.model = f"openai:{model}", model
        self._client = openai.OpenAI(timeout=settings.llm_timeout_s)

    def complete(self, system, user, *, json_mode=False, max_tokens=700, temperature=0.3) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        if json_mode:
            system += "\nRespond with a single JSON object only."
        with span("llm", self.name, prompt_chars=len(system) + len(user)) as s:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    **kwargs)
            except Exception as exc:
                raise LLMError(str(exc)) from exc
            if resp.usage:
                s["attrs"]["tokens"] = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
            return resp.choices[0].message.content or ""


@lru_cache(maxsize=1)
def get_llm() -> LLMClient | None:
    provider = settings.llm_provider
    if provider == "anthropic":
        return AnthropicClient(settings.anthropic_model)
    if provider == "openai":
        return OpenAIClient(settings.openai_model)
    return None


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise LLMError(f"No JSON object in model output: {text[:120]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON from model: {exc}") from exc

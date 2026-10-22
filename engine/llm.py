"""LLM provider interface, mock provider, KB retrieval and audited structured calls."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel
from sqlmodel import Session

from config import Settings
from database.db import append_audit
from engine.schemas import PolicyDecision, TicketSchema

T = TypeVar("T", bound=BaseModel)

TASK_EXTRACT = "extract"
TASK_DRAFT = "draft"
MAX_ATTEMPTS = 2  # first try + one retry


class LLMProviderError(Exception):
    """Provider could not produce any output (missing fixture, network error, ...)."""


class LLMOutputError(Exception):
    """Output still failed validation after the retry."""


@dataclass(frozen=True)
class LLMRequest:
    task: str
    prompt: str
    ticket_text: str  # the mock keys fixtures on this; real providers use only the prompt
    attempt: int


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, request: LLMRequest) -> str:
        """Return raw text that should be JSON for the task's schema."""


class MockProvider:
    """Deterministic fixtures from data/fixtures/*.json keyed by exact ticket text."""

    name = "mock"

    def __init__(self, fixtures_dir: Path, model: str = "mock-fixtures") -> None:
        self.model = model
        self._by_text: dict[str, dict[str, Any]] = {}
        for path in sorted(fixtures_dir.glob("*.json")):
            fixture = json.loads(path.read_text())
            self._by_text[fixture["ticket_text"].strip()] = fixture

    def generate(self, request: LLMRequest) -> str:
        fixture = self._by_text.get(request.ticket_text.strip())
        if fixture is None or request.task not in fixture:
            raise LLMProviderError(f"no mock fixture for task '{request.task}'")
        responses = fixture[request.task]
        if not isinstance(responses, list):
            responses = [responses]
        item = responses[min(request.attempt, len(responses) - 1)]
        return item if isinstance(item, str) else json.dumps(item)


MAX_OUTPUT_TOKENS = 4096
OLLAMA_TIMEOUT_SECONDS = 120


def _strip_code_fence(text: str) -> str:
    """Models sometimes wrap JSON in a ```json fence despite instructions."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].removesuffix("```")
    return stripped.strip()


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, client: Any = None) -> None:
        self.model = model
        if client is None:
            import anthropic  # lazy: mock mode needs no SDK

            client = anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY / profile
        self._client = client

    def generate(self, request: LLMRequest) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": request.prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMProviderError("model refused the request")
        text = "".join(b.text for b in response.content if b.type == "text")
        return _strip_code_fence(text)


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None, client: Any = None) -> None:
        self.model = model
        if client is None:
            import openai

            client = openai.OpenAI(api_key=api_key)
        self._client = client

    def generate(self, request: LLMRequest) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": request.prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("empty completion")
        return _strip_code_fence(content)


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, base_url: str, client: Any = None) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=OLLAMA_TIMEOUT_SECONDS)

    def generate(self, request: LLMRequest) -> str:
        response = self._client.post(
            f"{self._base_url}/api/chat",
            json={"model": self.model, "stream": False, "format": "json",
                  "messages": [{"role": "user", "content": request.prompt}]},
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        if not content:
            raise LLMProviderError("empty completion")
        return _strip_code_fence(content)


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value is not None else None


def get_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockProvider(settings.data_dir / "fixtures", settings.llm_model or "mock-fixtures")
    if not settings.llm_model:
        raise ValueError(f"LLM_MODEL must be set when LLM_PROVIDER={settings.llm_provider}")
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings.llm_model, _secret(settings.anthropic_api_key))
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings.llm_model, _secret(settings.openai_api_key))
    return OllamaProvider(settings.llm_model, settings.ollama_base_url)


# --- Knowledge base -----------------------------------------------------------
def parse_kb(markdown: str) -> dict[str, str]:
    """Split policy_kb.md into {section title lowercased: body} on '## ' headings."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = sections.setdefault(line[3:].strip().lower(), [])
        elif current is not None:
            current.append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def kb_section(kb_markdown: str, category: str) -> str:
    """Empty string means the KB does not cover this category."""
    return parse_kb(kb_markdown).get(category.lower(), "")


# --- Prompts ------------------------------------------------------------------
def build_extraction_prompt(ticket_text: str) -> str:
    schema = json.dumps(TicketSchema.model_json_schema())
    return (
        "Extract structured fields from the support ticket. Respond with JSON only, matching this "
        f"JSON schema:\n{schema}\nDo not decide policy or compute anything.\n\nTICKET:\n{ticket_text}"
    )


def build_draft_prompt(ticket_text: str, ticket: TicketSchema, decision: PolicyDecision, kb: str) -> str:
    return (
        "Draft a reply to the customer and propose an action_type. Use ONLY the knowledge-base excerpt "
        "below; if it does not answer the request, say so and choose REQUIRES_HUMAN_APPROVAL. "
        f"Allowed actions: {[a.value for a in decision.allowed_actions]}. Respond with JSON only "
        "with keys action_type, reasoning, draft_text.\n\n"
        f"KNOWLEDGE BASE:\n{kb}\n\nTICKET FIELDS:\n{ticket.model_dump_json()}\n\nTICKET:\n{ticket_text}"
    )


# --- Audited structured call --------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def structured_call(
    provider: LLMProvider,
    session: Session,
    *,
    task: str,
    prompt: str,
    ticket_text: str,
    schema: type[T],
    ticket_id: int | None,
) -> T:
    """Call the LLM, validate against `schema`, retry once feeding the error back.

    Every attempt appends an LLM_CALL audit row. Raises LLMOutputError if both fail;
    the caller owns the fail-safe routing.
    """
    current_prompt, error = prompt, ""
    for attempt in range(MAX_ATTEMPTS):
        raw, result = "", None
        try:
            raw = provider.generate(LLMRequest(task, current_prompt, ticket_text, attempt))
            result = schema.model_validate_json(raw)
        except Exception as exc:  # any provider/validation failure is a failed call (fail safe)
            error = str(exc)
        append_audit(
            session,
            event_type="LLM_CALL",
            actor=f"llm:{provider.name}",
            ticket_id=ticket_id,
            payload={
                "task": task,
                "provider": provider.name,
                "model": provider.model,
                "prompt_hash": _sha(current_prompt),
                "output_hash": _sha(raw),
                "attempt": attempt + 1,
                "outcome": "validated" if result is not None else "failed",
                "error": None if result is not None else error[:500],
            },
        )
        if result is not None:
            return result
        current_prompt = (
            f"{prompt}\n\nYour previous output failed validation with this error:\n{error[:2000]}\n"
            "Return corrected JSON only."
        )
    raise LLMOutputError(error)

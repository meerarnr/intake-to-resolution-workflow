import json

import pytest
from pydantic import BaseModel

from engine.llm import LLMOutputError, LLMProviderError, LLMRequest, MockProvider, kb_section, parse_kb, structured_call
from engine.schemas import TicketSchema
from tests.conftest import FIXTURES, load_fixture


class Scripted:
    """Provider returning scripted outputs; records prompts."""
    name, model = "scripted", "test-model"

    def __init__(self, outputs):
        self.outputs, self.prompts = outputs, []

    def generate(self, request: LLMRequest) -> str:
        self.prompts.append(request.prompt)
        return self.outputs[min(request.attempt, len(self.outputs) - 1)]


class Tiny(BaseModel):
    n: int


def audit_rows(session):
    from sqlmodel import select
    from database.models import AuditLog
    return [json.loads(r.payload_json) for r in session.exec(select(AuditLog).order_by(AuditLog.id))]


def test_retry_feeds_validation_error_back(session):
    p = Scripted(['{"n": "x"}', '{"n": 3}'])
    out = structured_call(p, session, task="t", prompt="P", ticket_text="", schema=Tiny, ticket_id=1)
    assert out.n == 3
    assert "failed validation" in p.prompts[1] and "int" in p.prompts[1]
    a = audit_rows(session)
    assert [x["outcome"] for x in a] == ["failed", "validated"]
    assert a[0]["prompt_hash"] != a[1]["prompt_hash"]
    assert {x["provider"] for x in a} == {"scripted"} and a[0]["model"] == "test-model"


def test_fails_after_exactly_one_retry(session):
    p = Scripted(["nope"])
    with pytest.raises(LLMOutputError):
        structured_call(p, session, task="t", prompt="P", ticket_text="", schema=Tiny, ticket_id=1)
    assert len(p.prompts) == 2
    assert [x["outcome"] for x in audit_rows(session)] == ["failed", "failed"]


def test_provider_error_counts_as_failed_call(session):
    class Boom(Scripted):
        def generate(self, request):
            raise LLMProviderError("down")
    with pytest.raises(LLMOutputError):
        structured_call(Boom([]), session, task="t", prompt="P", ticket_text="", schema=Tiny, ticket_id=1)


def test_mock_is_deterministic_and_keyed_by_text():
    m = MockProvider(FIXTURES)
    fx = load_fixture("known_resolution_technical")
    req = LLMRequest("extract", "ignored", fx["ticket_text"], 0)
    assert m.generate(req) == m.generate(req)
    TicketSchema.model_validate_json(m.generate(req))
    with pytest.raises(LLMProviderError):
        m.generate(LLMRequest("extract", "p", "unknown text", 0))


def test_all_fixtures_have_unique_text():
    texts = [json.loads(p.read_text())["ticket_text"] for p in FIXTURES.glob("*.json")]
    assert len(texts) == len(set(texts)) == 7


def test_kb_sections():
    kb = "# T\n\n## Billing\n- a\n\n## Technical\n- b\n"
    assert parse_kb(kb) == {"billing": "- a", "technical": "- b"}
    assert kb_section(kb, "Billing") == "- a"
    assert kb_section(kb, "Account") == ""


# --- real providers (fake SDK clients; no network) -------------------------------
from types import SimpleNamespace

import httpx
import pytest

from config import Settings
from engine.llm import AnthropicProvider, LLMProviderError, LLMRequest, OllamaProvider, OpenAIProvider, get_provider

REQ = LLMRequest(task="extract", prompt="PROMPT", ticket_text="t", attempt=0)


class FakeAnthropic:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.calls = []
        self._resp = SimpleNamespace(content=blocks, stop_reason=stop_reason)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self._resp


def test_anthropic_provider_returns_text_and_uses_env_model():
    fake = FakeAnthropic([SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text='```json\n{"a": 1}\n```')])
    out = AnthropicProvider("some-model", client=fake).generate(REQ)
    assert out == '{"a": 1}'
    assert fake.calls[0]["model"] == "some-model"
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "PROMPT"}]


def test_anthropic_refusal_is_a_provider_error():
    fake = FakeAnthropic([], stop_reason="refusal")
    with pytest.raises(LLMProviderError):
        AnthropicProvider("m", client=fake).generate(REQ)


def test_openai_provider_and_empty_completion():
    def make(content):
        msg = SimpleNamespace(content=content)
        create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)])  # noqa: E731
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    assert OpenAIProvider("m", client=make('{"a": 1}')).generate(REQ) == '{"a": 1}'
    with pytest.raises(LLMProviderError):
        OpenAIProvider("m", client=make(None)).generate(REQ)


def test_ollama_provider_posts_to_chat_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), request.read()
        return httpx.Response(200, json={"message": {"content": '{"a": 1}'}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = OllamaProvider("llama-x", "http://host:11434/", client=client).generate(REQ)
    assert out == '{"a": 1}' and seen["url"] == "http://host:11434/api/chat"
    assert b'"llama-x"' in seen["body"]


def test_ollama_http_error_propagates_for_failsafe():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(httpx.HTTPStatusError):
        OllamaProvider("m", "http://h", client=client).generate(REQ)


def test_get_provider_requires_model_for_real_providers():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        get_provider(Settings(llm_provider="openai", llm_model="", _env_file=None))


def test_get_provider_selects_by_env():
    assert get_provider(Settings(llm_provider="mock", _env_file=None)).name == "mock"
    assert get_provider(Settings(llm_provider="ollama", llm_model="m", _env_file=None)).name == "ollama"
    assert get_provider(Settings(llm_provider="openai", llm_model="m", openai_api_key="k", _env_file=None)).name == "openai"
    assert get_provider(Settings(llm_provider="anthropic", llm_model="m", anthropic_api_key="k", _env_file=None)).name == "anthropic"

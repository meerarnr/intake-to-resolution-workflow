from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlmodel import Session

from config import load_customers, load_known_resolution_codes, load_policy_rules
from database.db import init_db, make_engine
from engine.schemas import TicketSchema

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CREATED_AT = datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def rules():
    return load_policy_rules(DATA_DIR)


@pytest.fixture(scope="session")
def customers():
    return load_customers(DATA_DIR)


@pytest.fixture(scope="session")
def known_codes():
    return load_known_resolution_codes(DATA_DIR)


@pytest.fixture
def make_ticket():
    def _make(**overrides) -> TicketSchema:
        base = dict(
            customer_id="C-2001",
            category="General",
            urgency="Medium",
            sentiment="Neutral",
            extracted_entities={},
            confidence_score=0.95,
            requested_refund_amount=None,
        )
        return TicketSchema(**{**base, **overrides})

    return _make


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as s:
        yield s


# --- workflow environment -----------------------------------------------------
import json
from dataclasses import dataclass

from config import load_policy_kb
from database.db import verify_chain
from database.models import AuditLog, Ticket
from engine.llm import MockProvider
from engine.workflow import WorkflowDeps, build_graph, create_ticket, open_checkpointer, start_ticket
from sqlmodel import select

FIXTURES = DATA_DIR / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@dataclass
class Env:
    deps: WorkflowDeps
    graph: object
    db_url: str
    cp_path: Path

    def run(self, fixture_name: str) -> int:
        fx = load_fixture(fixture_name)
        tid = create_ticket(self.deps, fx["customer_id"], fx["ticket_text"])
        start_ticket(self.graph, tid)
        return tid

    def status(self, tid: int) -> str:
        with Session(self.deps.engine) as s:
            return s.get(Ticket, tid).status.value

    def audit(self, tid: int | None = None) -> list[AuditLog]:
        with Session(self.deps.engine) as s:
            rows = s.exec(select(AuditLog).order_by(AuditLog.id)).all()
        return [r for r in rows if tid is None or r.ticket_id == tid]

    def events(self, tid: int) -> list[str]:
        return [r.event_type for r in self.audit(tid)]

    def assert_chain_valid(self) -> None:
        with Session(self.deps.engine) as s:
            result = verify_chain(s)
        assert result.valid, result

    def restart(self) -> "Env":
        """Simulate a process restart: brand-new engine, provider, checkpointer connection and graph."""
        self.graph.checkpointer.conn.close()
        return make_env(self.db_url, self.cp_path, kb=self.deps.kb)


def make_env(db_url: str, cp_path: Path, kb: str | None = None) -> Env:
    engine = make_engine(db_url)
    init_db(engine)
    deps = WorkflowDeps(
        engine=engine,
        provider=MockProvider(FIXTURES),
        rules=load_policy_rules(DATA_DIR),
        customers=load_customers(DATA_DIR),
        known_codes=load_known_resolution_codes(DATA_DIR),
        kb=kb if kb is not None else load_policy_kb(DATA_DIR),
    )
    graph = build_graph(deps, open_checkpointer(cp_path))
    return Env(deps, graph, db_url, cp_path)


@pytest.fixture
def env(tmp_path):
    e = make_env(f"sqlite:///{tmp_path / 'wf.db'}", tmp_path / "checkpoints.db")
    yield e
    e.graph.checkpointer.conn.close()

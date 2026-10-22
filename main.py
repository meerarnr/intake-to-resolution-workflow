"""FastAPI surface. Only the endpoints in the spec; all state changes go through the workflow."""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlmodel import Session, select

from config import (
    Settings,
    get_settings,
    load_customers,
    load_known_resolution_codes,
    load_policy_kb,
    load_policy_rules,
)
from database.db import init_db, make_engine, verify_chain
from database.models import AuditLog, Ticket
from engine.llm import get_provider
from engine.schemas import (
    Approver,
    ClassificationOverride,
    DraftResponseAction,
    HumanAction,
    HumanDecision,
    NonEmptyStr,
    PolicyDecision,
    TicketStatus,
)
from engine.workflow import (
    WorkflowDeps,
    as_utc,
    build_graph,
    create_ticket,
    is_paused,
    open_checkpointer,
    resume_ticket,
    start_ticket,
)


# --- request / response models --------------------------------------------------
class TicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: NonEmptyStr
    text: NonEmptyStr = Field(max_length=20_000)


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: NonEmptyStr
    reason: NonEmptyStr
    edited_draft: NonEmptyStr | None = None
    # Self-asserted: the project has no auth. Used only for the manager-approval privilege check.
    approver_role: Approver = Approver.SUPPORT_AGENT


class RejectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: NonEmptyStr
    reason: NonEmptyStr
    action: Literal["override", "close"]
    override: ClassificationOverride | None = None
    approver_role: Approver = Approver.SUPPORT_AGENT

    @model_validator(mode="after")
    def _override_matches_action(self) -> "RejectRequest":
        if self.action == "override" and (self.override is None or not self.override.model_fields_set):
            raise ValueError("action 'override' requires at least one override field")
        if self.action == "close" and self.override is not None:
            raise ValueError("action 'close' must not include an override")
        return self


class TicketSummary(BaseModel):
    id: int
    customer_id: str
    status: TicketStatus
    category: str | None
    urgency: str | None
    sentiment: str | None
    created_at: datetime
    updated_at: datetime


class TicketDetail(TicketSummary):
    raw_text: str
    confidence_score: float | None
    requested_refund_amount: Decimal | None
    awaiting_human: bool
    final_action: str | None
    failsafe_reason: str | None
    policy_decision: PolicyDecision | None
    draft: DraftResponseAction | None


class AuditEntry(BaseModel):
    id: int
    ticket_id: int | None
    event_type: str
    actor: str
    timestamp: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str


class ChainStatus(BaseModel):
    valid: bool
    checked: int
    first_bad_id: int | None


# --- app --------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        cfg = settings or get_settings()
        engine = make_engine(cfg.database_url)
        init_db(engine)
        deps = WorkflowDeps(
            engine=engine,
            provider=get_provider(cfg),
            rules=load_policy_rules(cfg.data_dir),
            customers=load_customers(cfg.data_dir),
            known_codes=load_known_resolution_codes(cfg.data_dir),
            kb=load_policy_kb(cfg.data_dir),
        )
        checkpointer = open_checkpointer(cfg.checkpoint_path)
        app.state.deps = deps
        app.state.graph = build_graph(deps, checkpointer)
        # Serializes workflow runs so concurrent requests cannot fork the audit hash chain.
        app.state.lock = threading.Lock()
        yield
        checkpointer.conn.close()
        engine.dispose()

    app = FastAPI(title="Intake-to-Resolution", lifespan=lifespan)

    def session(request: Request) -> Session:
        return Session(request.app.state.deps.engine, expire_on_commit=False)

    def load(s: Session, ticket_id: int) -> Ticket:
        ticket = s.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(404, f"ticket {ticket_id} not found")
        return ticket

    def summary_fields(t: Ticket) -> dict[str, Any]:
        return dict(
            id=t.id, customer_id=t.customer_id, status=t.status, category=t.category, urgency=t.urgency,
            sentiment=t.sentiment, created_at=as_utc(t.created_at), updated_at=as_utc(t.updated_at),
        )

    def detail(request: Request, t: Ticket) -> TicketDetail:
        graph = request.app.state.graph
        values = graph.get_state({"configurable": {"thread_id": f"ticket-{t.id}"}}).values or {}
        return TicketDetail(
            **summary_fields(t),
            raw_text=t.raw_text,
            confidence_score=t.confidence_score,
            requested_refund_amount=t.requested_refund_amount,
            awaiting_human=t.status is TicketStatus.AWAITING_HUMAN_ACTION,
            final_action=values.get("final_action"),
            failsafe_reason=values.get("failsafe_reason"),
            policy_decision=values.get("decision"),
            draft=values.get("draft"),
        )

    def act(request: Request, ticket_id: int, human: HumanDecision) -> TicketDetail:
        """Shared approve/reject path: 404, 409 if not awaiting, 403/422 on bad input, then resume."""
        graph = request.app.state.graph
        with request.app.state.lock:
            with session(request) as s:
                t = load(s, ticket_id)
                if t.status is not TicketStatus.AWAITING_HUMAN_ACTION or not is_paused(graph, ticket_id):
                    raise HTTPException(409, f"ticket {ticket_id} is {t.status.value}, not awaiting human action")
            try:
                resume_ticket(graph, ticket_id, human)
            except PermissionError as exc:
                raise HTTPException(403, str(exc)) from exc
            except ValidationError as exc:
                raise HTTPException(422, json.loads(exc.json(include_url=False))) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            with session(request) as s:
                return detail(request, load(s, ticket_id))

    @app.post("/tickets", response_model=TicketDetail, status_code=201)
    def submit_ticket(body: TicketCreate, request: Request) -> TicketDetail:
        deps: WorkflowDeps = request.app.state.deps
        with request.app.state.lock:
            ticket_id = create_ticket(deps, body.customer_id, body.text)
            start_ticket(request.app.state.graph, ticket_id)
            with session(request) as s:
                return detail(request, load(s, ticket_id))

    @app.get("/tickets", response_model=list[TicketSummary])
    def list_tickets(request: Request) -> list[TicketSummary]:
        with session(request) as s:
            rows = s.exec(select(Ticket).order_by(Ticket.id)).all()  # type: ignore[arg-type]
            return [TicketSummary(**summary_fields(t)) for t in rows]

    @app.get("/tickets/{ticket_id}", response_model=TicketDetail)
    def get_ticket(ticket_id: int, request: Request) -> TicketDetail:
        with session(request) as s:
            return detail(request, load(s, ticket_id))

    @app.get("/queue", response_model=list[TicketDetail])
    def queue(request: Request) -> list[TicketDetail]:
        with session(request) as s:
            rows = s.exec(
                select(Ticket).where(Ticket.status == TicketStatus.AWAITING_HUMAN_ACTION).order_by(Ticket.id)  # type: ignore[arg-type]
            ).all()
            return [detail(request, t) for t in rows]

    @app.post("/tickets/{ticket_id}/approve", response_model=TicketDetail)
    def approve(ticket_id: int, body: ApproveRequest, request: Request) -> TicketDetail:
        human = HumanDecision(
            decision=HumanAction.APPROVE, reviewer=body.reviewer, reason=body.reason,
            edited_draft=body.edited_draft, approver_role=body.approver_role,
        )
        return act(request, ticket_id, human)

    @app.post("/tickets/{ticket_id}/reject", response_model=TicketDetail)
    def reject(ticket_id: int, body: RejectRequest, request: Request) -> TicketDetail:
        human = HumanDecision(
            decision=HumanAction.REJECT_OVERRIDE if body.action == "override" else HumanAction.REJECT_CLOSE,
            reviewer=body.reviewer, reason=body.reason, override=body.override, approver_role=body.approver_role,
        )
        return act(request, ticket_id, human)

    @app.get("/tickets/{ticket_id}/audit", response_model=list[AuditEntry])
    def ticket_audit(ticket_id: int, request: Request) -> list[AuditEntry]:
        with session(request) as s:
            load(s, ticket_id)
            rows = s.exec(select(AuditLog).where(AuditLog.ticket_id == ticket_id).order_by(AuditLog.id)).all()  # type: ignore[arg-type]
            return [
                AuditEntry(
                    id=r.id, ticket_id=r.ticket_id, event_type=r.event_type, actor=r.actor, timestamp=r.timestamp,
                    payload=json.loads(r.payload_json), prev_hash=r.prev_hash, entry_hash=r.entry_hash,
                )
                for r in rows
            ]

    @app.get("/audit/verify", response_model=ChainStatus)
    def audit_verify(request: Request) -> ChainStatus:
        with session(request) as s:
            result = verify_chain(s)
        return ChainStatus(valid=result.valid, checked=result.checked, first_bad_id=result.first_bad_id)

    return app


app = create_app()

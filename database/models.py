"""SQLModel tables. AuditLog is append-only; UPDATE/DELETE are blocked at the ORM layer."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import event
from sqlalchemy.orm import Session as SASession
from sqlmodel import Field, SQLModel

from engine.schemas import TicketStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ticket(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    customer_id: str = Field(index=True)
    raw_text: str
    status: TicketStatus = Field(default=TicketStatus.NEW, index=True)
    # Structured fields are filled after LLM extraction. Refund is a str to keep Decimal exact.
    category: str | None = None
    urgency: str | None = None
    sentiment: str | None = None
    confidence_score: float | None = None
    requested_refund_amount: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class PolicyRule(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    rule_id: str = Field(unique=True, index=True)
    name: str
    precedence: int
    description: str = ""
    active: bool = True


class StateSnapshot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ticket_id: int = Field(foreign_key="ticket.id", index=True)
    step: str
    state_json: str
    created_at: datetime = Field(default_factory=utcnow)


class AuditLog(SQLModel, table=True):
    """One row per state transition, LLM call, policy check or human decision.

    entry_hash = SHA-256(canonical content + prev_hash); ordered by id.
    Written only via database.db.append_audit.
    """

    id: int | None = Field(default=None, primary_key=True)
    ticket_id: int | None = Field(default=None, index=True)
    event_type: str
    actor: str
    payload_json: str  # canonical JSON, stored as hashed
    timestamp: str  # ISO-8601 UTC string, stored as hashed
    prev_hash: str
    entry_hash: str = Field(index=True)


class AuditLogImmutableError(Exception):
    """Raised on any attempt to UPDATE or DELETE an AuditLog row."""


@event.listens_for(AuditLog, "before_update")
def _block_update(*_: object) -> None:
    raise AuditLogImmutableError("AuditLog rows cannot be updated")


@event.listens_for(AuditLog, "before_delete")
def _block_delete(*_: object) -> None:
    raise AuditLogImmutableError("AuditLog rows cannot be deleted")


@event.listens_for(SASession, "do_orm_execute")
def _block_bulk_writes(state) -> None:  # type: ignore[no-untyped-def]
    # Covers session.execute(update(AuditLog)...) / delete(AuditLog)..., which skip mapper events.
    if (state.is_update or state.is_delete) and any(
        m.class_ is AuditLog for m in state.all_mappers
    ):
        raise AuditLogImmutableError("AuditLog rows cannot be updated or deleted")

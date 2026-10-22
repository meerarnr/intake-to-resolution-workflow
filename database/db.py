"""Engine/session helpers and the only audit write path."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from database.models import AuditLog, utcnow

GENESIS_HASH = "0" * 64


def make_engine(url: str) -> Engine:
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_entry_hash(
    *, ticket_id: int | None, event_type: str, actor: str, payload_json: str, timestamp: str, prev_hash: str
) -> str:
    content = _canonical(
        {
            "ticket_id": ticket_id,
            "event_type": event_type,
            "actor": actor,
            "payload_json": payload_json,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
        }
    )
    return hashlib.sha256(content.encode()).hexdigest()


def append_audit(
    session: Session,
    *,
    event_type: str,
    actor: str,
    payload: dict[str, Any],
    ticket_id: int | None = None,
) -> AuditLog:
    """Append one chained row. Flushes but does not commit, so the caller's transaction owns it."""
    last = session.exec(select(AuditLog).order_by(AuditLog.id.desc()).limit(1)).first()  # type: ignore[union-attr]
    prev_hash = last.entry_hash if last else GENESIS_HASH
    payload_json = _canonical(payload)
    timestamp = utcnow().isoformat()
    row = AuditLog(
        ticket_id=ticket_id,
        event_type=event_type,
        actor=actor,
        payload_json=payload_json,
        timestamp=timestamp,
        prev_hash=prev_hash,
        entry_hash=compute_entry_hash(
            ticket_id=ticket_id,
            event_type=event_type,
            actor=actor,
            payload_json=payload_json,
            timestamp=timestamp,
            prev_hash=prev_hash,
        ),
    )
    session.add(row)
    session.flush()
    return row


@dataclass(frozen=True)
class ChainVerification:
    valid: bool
    checked: int
    first_bad_id: int | None = None


def verify_chain(session: Session) -> ChainVerification:
    """Recompute every hash in id order; report the first row that doesn't match."""
    prev_hash = GENESIS_HASH
    checked = 0
    for row in session.exec(select(AuditLog).order_by(AuditLog.id)):  # type: ignore[arg-type]
        expected = compute_entry_hash(
            ticket_id=row.ticket_id,
            event_type=row.event_type,
            actor=row.actor,
            payload_json=row.payload_json,
            timestamp=row.timestamp,
            prev_hash=row.prev_hash,
        )
        if row.prev_hash != prev_hash or row.entry_hash != expected:
            return ChainVerification(False, checked, row.id)
        prev_hash = row.entry_hash
        checked += 1
    return ChainVerification(True, checked)

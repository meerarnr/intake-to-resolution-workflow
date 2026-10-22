import pytest
from sqlalchemy import delete, text, update
from sqlmodel import select

from database.db import GENESIS_HASH, append_audit, verify_chain
from database.models import AuditLog, AuditLogImmutableError


@pytest.fixture
def populated(session):
    for i in range(4):
        append_audit(session, event_type="STATE_TRANSITION", actor="system", ticket_id=1, payload={"step": i})
    session.commit()
    return session


def test_empty_chain_valid(session):
    assert verify_chain(session).valid


def test_chain_links_and_verifies(populated):
    rows = populated.exec(select(AuditLog).order_by(AuditLog.id)).all()
    assert rows[0].prev_hash == GENESIS_HASH
    assert all(b.prev_hash == a.entry_hash for a, b in zip(rows, rows[1:]))
    result = verify_chain(populated)
    assert result.valid and result.checked == 4


def test_tampered_payload_detected(populated):
    populated.connection().execute(
        text("UPDATE auditlog SET payload_json=:p WHERE id=2"), {"p": '{"step":99}'}
    )
    populated.commit()
    result = verify_chain(populated)
    assert not result.valid and result.first_bad_id == 2


def test_tampered_hash_with_recompute_still_breaks_next_link(populated):
    # Attacker rewrites row 2's hash to match edited content; row 3's prev_hash no longer matches.
    from database.db import compute_entry_hash
    row = populated.get(AuditLog, 2)
    new_payload = '{"step":99}'
    new_hash = compute_entry_hash(
        ticket_id=row.ticket_id, event_type=row.event_type, actor=row.actor,
        payload_json=new_payload, timestamp=row.timestamp, prev_hash=row.prev_hash,
    )
    populated.connection().execute(
        text("UPDATE auditlog SET payload_json=:p, entry_hash=:h WHERE id=2"), {"p": new_payload, "h": new_hash}
    )
    populated.commit()
    result = verify_chain(populated)
    assert not result.valid and result.first_bad_id == 3


def test_deleted_middle_row_detected(populated):
    populated.connection().execute(text("DELETE FROM auditlog WHERE id=2"))
    populated.commit()
    result = verify_chain(populated)
    assert not result.valid and result.first_bad_id == 3


def test_orm_update_rejected(populated):
    row = populated.get(AuditLog, 1)
    row.actor = "attacker"
    with pytest.raises(AuditLogImmutableError):
        populated.commit()
    populated.rollback()
    assert populated.get(AuditLog, 1).actor == "system"


def test_orm_delete_rejected(populated):
    populated.delete(populated.get(AuditLog, 1))
    with pytest.raises(AuditLogImmutableError):
        populated.commit()
    populated.rollback()
    assert len(populated.exec(select(AuditLog)).all()) == 4


def test_bulk_update_and_delete_rejected(populated):
    with pytest.raises(AuditLogImmutableError):
        populated.execute(update(AuditLog).values(actor="attacker"))
    with pytest.raises(AuditLogImmutableError):
        populated.execute(delete(AuditLog))
    assert verify_chain(populated).valid

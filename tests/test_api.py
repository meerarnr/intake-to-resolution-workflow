import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import Session

from config import Settings
from database.db import make_engine
from main import create_app
from tests.conftest import DATA_DIR, load_fixture


def settings_for(tmp_path) -> Settings:
    return Settings(
        llm_provider="mock", data_dir=DATA_DIR,
        database_url=f"sqlite:///{tmp_path / 'api.db'}", checkpoint_path=tmp_path / "cp.db",
    )


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as c:
        yield c


def submit(client, fixture):
    fx = load_fixture(fixture)
    r = client.post("/tickets", json={"customer_id": fx["customer_id"], "text": fx["ticket_text"]})
    return r


def approve_body(**kw):
    return {"reviewer": "alice", "reason": "looks right", "approver_role": "MANAGER", **kw}


# --- POST /tickets ------------------------------------------------------------
def test_submit_auto_resolves(client):
    r = submit(client, "known_resolution_technical")
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "RESOLVED" and not body["awaiting_human"]
    assert body["category"] == "Technical" and body["policy_decision"]["triggered_rules"] == ["RULE_3_AUTO_RESOLVE"]


def test_submit_pauses_for_human(client):
    body = submit(client, "billing_refund_over_250").json()
    assert body["status"] == "AWAITING_HUMAN_ACTION" and body["awaiting_human"]
    assert body["requested_refund_amount"] == "480.00"
    assert body["policy_decision"]["required_approver"] == "MANAGER"
    assert body["draft"]["draft_text"]


def test_submit_unknown_text_fails_safe_to_queue(client):
    r = client.post("/tickets", json={"customer_id": "C-2001", "text": "text with no mock fixture"})
    assert r.status_code == 201
    assert r.json()["status"] == "AWAITING_HUMAN_ACTION" and r.json()["failsafe_reason"] == "extraction_failed"


@pytest.mark.parametrize("body", [
    {}, {"customer_id": "C-1"}, {"customer_id": " ", "text": "x"}, {"customer_id": "C-1", "text": "x", "extra": 1},
])
def test_submit_validation_422(client, body):
    assert client.post("/tickets", json=body).status_code == 422


# --- GET /tickets, /tickets/{id}, /queue --------------------------------------
def test_list_and_detail(client):
    a = submit(client, "known_resolution_technical").json()["id"]
    b = submit(client, "billing_refund_over_250").json()["id"]
    rows = client.get("/tickets").json()
    assert [r["id"] for r in rows] == [a, b] and "raw_text" not in rows[0]
    d = client.get(f"/tickets/{b}").json()
    assert d["id"] == b and d["raw_text"] and d["status"] == "AWAITING_HUMAN_ACTION"


def test_detail_404(client):
    assert client.get("/tickets/999").status_code == 404


def test_queue_only_awaiting(client):
    submit(client, "known_resolution_technical")
    b = submit(client, "billing_refund_over_250").json()["id"]
    c = submit(client, "legal_terms").json()["id"]
    assert [t["id"] for t in client.get("/queue").json()] == [b, c]
    client.post(f"/tickets/{b}/approve", json=approve_body(edited_draft="ok"))
    assert [t["id"] for t in client.get("/queue").json()] == [c]


# --- approve --------------------------------------------------------------------
def test_approve_with_edit(client):
    tid = submit(client, "billing_refund_over_250").json()["id"]
    r = client.post(f"/tickets/{tid}/approve", json=approve_body(edited_draft="Refund approved."))
    assert r.status_code == 200
    assert r.json()["status"] == "RESOLVED" and r.json()["draft"]["draft_text"] == "Refund approved."
    h = [e for e in client.get(f"/tickets/{tid}/audit").json() if e["event_type"] == "HUMAN_DECISION"][0]
    assert h["actor"] == "human:alice" and h["payload"]["reason"] == "looks right" and h["payload"]["draft_edited"]


def test_approve_without_edit(client):
    tid = submit(client, "billing_refund_over_250").json()["id"]
    r = client.post(f"/tickets/{tid}/approve", json=approve_body())
    assert r.status_code == 200 and r.json()["status"] == "RESOLVED"


@pytest.mark.parametrize("body", [{}, {"reviewer": "a"}, {"reason": "r"}, {"reviewer": " ", "reason": "r"}])
def test_approve_requires_reviewer_and_reason(client, body):
    tid = submit(client, "billing_refund_over_250").json()["id"]
    assert client.post(f"/tickets/{tid}/approve", json=body).status_code == 422
    assert client.get(f"/tickets/{tid}").json()["status"] == "AWAITING_HUMAN_ACTION"


def test_approve_insufficient_role_403(client):
    tid = submit(client, "billing_refund_over_250").json()["id"]
    r = client.post(f"/tickets/{tid}/approve", json=approve_body(approver_role="SUPPORT_AGENT"))
    assert r.status_code == 403
    assert client.get(f"/tickets/{tid}").json()["status"] == "AWAITING_HUMAN_ACTION"


def test_approve_failsafe_needs_edited_draft_422(client):
    tid = client.post("/tickets", json={"customer_id": "C-2001", "text": "no fixture for this"}).json()["id"]
    assert client.post(f"/tickets/{tid}/approve", json=approve_body()).status_code == 422
    assert client.post(f"/tickets/{tid}/approve", json=approve_body(edited_draft="Hello")).status_code == 200


# --- reject ---------------------------------------------------------------------
def test_reject_override_reruns(client):
    tid = submit(client, "low_confidence_vague").json()["id"]
    body = {"reviewer": "bob", "reason": "this is a cache issue", "action": "override",
            "override": {"category": "Technical", "extracted_entities": {"resolution_code": "RES-CLEAR-CACHE"}}}
    r = client.post(f"/tickets/{tid}/reject", json=body)
    assert r.status_code == 200 and r.json()["status"] == "RESOLVED"
    audit = client.get(f"/tickets/{tid}/audit").json()
    assert [e["event_type"] for e in audit].count("POLICY_CHECK") == 2
    h = [e for e in audit if e["event_type"] == "HUMAN_DECISION"][0]
    assert h["payload"]["reason"] == "this is a cache issue" and h["payload"]["override"]["category"] == "Technical"


def test_reject_close(client):
    tid = submit(client, "billing_refund_over_250").json()["id"]
    r = client.post(f"/tickets/{tid}/reject", json={"reviewer": "bob", "reason": "duplicate", "action": "close"})
    assert r.status_code == 200 and r.json()["status"] == "CLOSED"


@pytest.mark.parametrize("body", [
    {"reviewer": "b", "reason": "r"},  # no action
    {"reviewer": "b", "reason": "r", "action": "override"},  # override missing
    {"reviewer": "b", "reason": "r", "action": "override", "override": {}},
    {"reviewer": "b", "reason": "r", "action": "close", "override": {"category": "Billing"}},
    {"reviewer": "b", "action": "close"},  # reason missing
    {"reason": "r", "action": "close"},  # reviewer missing
    {"reviewer": "b", "reason": "r", "action": "override", "override": {"urgency": "Urgent"}},
])
def test_reject_validation_422(client, body):
    tid = submit(client, "billing_refund_over_250").json()["id"]
    assert client.post(f"/tickets/{tid}/reject", json=body).status_code == 422
    assert client.get(f"/tickets/{tid}").json()["status"] == "AWAITING_HUMAN_ACTION"


def test_reject_override_incomplete_after_failed_extraction_422(client):
    tid = client.post("/tickets", json={"customer_id": "C-2001", "text": "no fixture here"}).json()["id"]
    r = client.post(f"/tickets/{tid}/reject", json={
        "reviewer": "b", "reason": "r", "action": "override", "override": {"category": "Billing"}})
    assert r.status_code == 422
    assert client.get(f"/tickets/{tid}").json()["status"] == "AWAITING_HUMAN_ACTION"


# --- 404 / 409 ------------------------------------------------------------------
@pytest.mark.parametrize("path,body", [
    ("approve", {"reviewer": "a", "reason": "r"}),
    ("reject", {"reviewer": "a", "reason": "r", "action": "close"}),
])
def test_act_on_missing_ticket_404(client, path, body):
    assert client.post(f"/tickets/12345/{path}", json=body).status_code == 404


@pytest.mark.parametrize("path,body", [
    ("approve", {"reviewer": "a", "reason": "r", "approver_role": "MANAGER"}),
    ("reject", {"reviewer": "a", "reason": "r", "action": "close"}),
])
@pytest.mark.parametrize("first", ["approve", "close", "auto"])
def test_act_when_not_awaiting_409(client, path, body, first):
    if first == "auto":
        tid = submit(client, "known_resolution_technical").json()["id"]
    else:
        tid = submit(client, "billing_refund_over_250").json()["id"]
        if first == "approve":
            client.post(f"/tickets/{tid}/approve", json=approve_body(edited_draft="x"))
        else:
            client.post(f"/tickets/{tid}/reject", json={"reviewer": "a", "reason": "r", "action": "close"})
    before = len(client.get(f"/tickets/{tid}/audit").json())
    assert client.post(f"/tickets/{tid}/{path}", json=body).status_code == 409
    assert len(client.get(f"/tickets/{tid}/audit").json()) == before  # nothing written


# --- audit ----------------------------------------------------------------------
def test_ticket_audit_trail(client):
    tid = submit(client, "known_resolution_technical").json()["id"]
    audit = client.get(f"/tickets/{tid}/audit").json()
    assert [e["event_type"] for e in audit][0] == "TICKET_CREATED"
    assert all(e["ticket_id"] == tid and len(e["entry_hash"]) == 64 for e in audit)
    assert isinstance(audit[0]["payload"], dict)
    llm = [e for e in audit if e["event_type"] == "LLM_CALL"][0]["payload"]
    assert llm["provider"] == "mock" and llm["prompt_hash"]


def test_ticket_audit_404(client):
    assert client.get("/tickets/999/audit").status_code == 404


def test_audit_verify_valid_then_tampered(client, tmp_path):
    submit(client, "billing_refund_over_250")
    r = client.get("/audit/verify")
    assert r.status_code == 200 and r.json()["valid"] and r.json()["checked"] > 0
    engine = make_engine(f"sqlite:///{tmp_path / 'api.db'}")
    with Session(engine) as s:
        s.connection().execute(text("UPDATE auditlog SET actor='mallory' WHERE id=2"))
        s.commit()
    bad = client.get("/audit/verify").json()
    assert bad["valid"] is False and bad["first_bad_id"] == 2


def test_audit_verify_empty(client):
    assert client.get("/audit/verify").json() == {"valid": True, "checked": 0, "first_bad_id": None}


# --- restart --------------------------------------------------------------------
def test_pending_ticket_survives_app_restart(tmp_path):
    cfg = settings_for(tmp_path)
    with TestClient(create_app(cfg)) as c1:
        tid = submit(c1, "billing_refund_over_250").json()["id"]
    with TestClient(create_app(cfg)) as c2:
        assert [t["id"] for t in c2.get("/queue").json()] == [tid]
        r = c2.post(f"/tickets/{tid}/approve", json=approve_body(edited_draft="after restart"))
        assert r.status_code == 200 and r.json()["status"] == "RESOLVED"
        assert c2.get("/audit/verify").json()["valid"]


def test_only_spec_endpoints_exist(client):
    routes = {(m, r.path) for r in client.app.routes if hasattr(r, "methods") for m in r.methods if m not in {"HEAD", "OPTIONS"}}
    app_routes = {x for x in routes if not x[1].startswith(("/docs", "/redoc", "/openapi"))}
    assert app_routes == {
        ("POST", "/tickets"), ("GET", "/tickets"), ("GET", "/tickets/{ticket_id}"), ("GET", "/queue"),
        ("POST", "/tickets/{ticket_id}/approve"), ("POST", "/tickets/{ticket_id}/reject"),
        ("GET", "/tickets/{ticket_id}/audit"), ("GET", "/audit/verify"),
    }

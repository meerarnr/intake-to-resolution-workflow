from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import app as dashboard
from main import create_app
from tests.test_api import settings_for

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def ticket(status, deadline_in_hours, updated=NOW):
    return {
        "status": status, "updated_at": updated.isoformat(),
        "policy_decision": {"sla_deadline": (NOW + timedelta(hours=deadline_in_hours)).isoformat()},
    }


def test_sla_countdown_for_open_ticket():
    assert dashboard.sla_label(ticket("AWAITING_HUMAN_ACTION", 1.5), NOW) == "1h 30m left"


def test_sla_breach_flag_for_open_ticket():
    assert dashboard.sla_label(ticket("AWAITING_HUMAN_ACTION", -2), NOW).startswith("BREACHED")


def test_sla_closed_ticket_met_or_breached():
    assert dashboard.sla_label(ticket("RESOLVED", 1, updated=NOW), NOW) == "met"
    assert dashboard.sla_label(ticket("RESOLVED", -1, updated=NOW), NOW) == "BREACHED"


def test_sla_without_decision():
    assert dashboard.sla_label({"status": "NEW", "policy_decision": None}, NOW) == "n/a"


def test_audit_summary_state_transition():
    entry = {"event_type": "STATE_TRANSITION", "payload": {"from": "NEW", "to": "PROCESSING", "reason": "start"}}
    assert dashboard.audit_summary(entry) == "NEW -> PROCESSING (start)"


def test_samples_cover_all_fixtures():
    assert "billing_refund_over_250" in dashboard.load_samples()


@pytest.fixture
def api(tmp_path, monkeypatch):
    with TestClient(create_app(settings_for(tmp_path))) as c:
        # Route the dashboard's httpx client into the in-process API.
        monkeypatch.setattr(httpx, "Client", lambda **kw: c)
        yield c


def run_dashboard() -> AppTest:
    return AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=30).run()


def test_dashboard_end_to_end(api):
    at = run_dashboard()
    assert not at.exception
    at.sidebar.selectbox[0].select("billing_refund_over_250")
    at.sidebar.button[0].click().run()
    assert not at.exception and not at.error
    assert api.get("/queue").json()[0]["category"] == "Billing"

    at.text_input(key="t1-rev").input("alice")
    at.text_input(key="t1-reason").input("verified duplicate charge")
    at.selectbox(key="t1-role").select("MANAGER")
    next(b for b in at.button if b.key == "t1-approve").click().run()
    assert not at.exception
    assert api.get("/tickets/1").json()["status"] == "RESOLVED"
    assert api.get("/audit/verify").json()["valid"]
    assert any("Chain valid" in s.value for s in at.success)


def test_dashboard_surfaces_api_error(api):
    at = run_dashboard()
    at.sidebar.selectbox[0].select("billing_refund_over_250")
    at.sidebar.button[0].click().run()
    # Missing reviewer/reason is rejected by the API; the dashboard shows it instead of crashing.
    next(b for b in at.button if b.key == "t1-approve").click().run()
    assert not at.exception
    assert at.error

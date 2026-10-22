"""Streamlit dashboard. Talks only to the FastAPI service (API_URL), never to the database."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
FIXTURES_DIR = Path(__file__).resolve().parent / "data" / "fixtures"
CATEGORIES = ["Billing", "Technical", "Account", "General"]
URGENCIES = ["Critical", "High", "Medium", "Low"]
SENTIMENTS = ["Positive", "Neutral", "Frustrated"]
ROLES = ["SUPPORT_AGENT", "MANAGER"]
UNCHANGED = "(unchanged)"
OPEN_STATUSES = {"NEW", "PROCESSING", "AWAITING_HUMAN_ACTION"}


# --- pure helpers ----------------------------------------------------------------
def sla_label(ticket: dict[str, Any], now: datetime | None = None) -> str:
    """Countdown or breach flag from the API-supplied deadline. Closed tickets compare against updated_at."""
    decision = ticket.get("policy_decision")
    if not decision:
        return "n/a"
    deadline = datetime.fromisoformat(decision["sla_deadline"])
    now = now or datetime.now(timezone.utc)
    if ticket["status"] in OPEN_STATUSES:
        left = deadline - now
        if left.total_seconds() < 0:
            return f"BREACHED ({_fmt_delta(-left.total_seconds())} over)"
        return f"{_fmt_delta(left.total_seconds())} left"
    finished = datetime.fromisoformat(ticket["updated_at"])
    return "met" if finished <= deadline else "BREACHED"


def _fmt_delta(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def audit_summary(entry: dict[str, Any]) -> str:
    """One-line description of an audit row for the timeline."""
    p, kind = entry["payload"], entry["event_type"]
    if kind == "STATE_TRANSITION":
        return f"{p.get('from')} -> {p.get('to')} ({p.get('reason', '')})"
    if kind == "DECISION":
        return f"final action {p.get('final_action')} ({p.get('basis')})"
    if kind == "POLICY_CHECK":
        rules = ", ".join(p.get("decision", {}).get("triggered_rules", [])) or "none"
        return f"triggered rules: {rules}"
    if kind == "HUMAN_DECISION":
        return f"{p.get('decision')}: {p.get('reason')}"
    return ", ".join(f"{k}={v}" for k, v in list(p.items())[:3])[:200]


def load_samples() -> dict[str, dict[str, str]]:
    samples = {}
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fx = json.loads(path.read_text())
        samples[fx["name"]] = {"customer_id": fx["customer_id"], "text": fx["ticket_text"]}
    return samples


# --- API access ------------------------------------------------------------------
def client() -> httpx.Client:
    if "http" not in st.session_state:
        st.session_state.http = httpx.Client(base_url=API_URL, timeout=30)
    return st.session_state.http


def call(method: str, path: str, **kwargs: Any) -> Any | None:
    """Return parsed JSON, or None after showing the API's error message."""
    try:
        resp = client().request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")
        return None
    if resp.is_error:
        st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
        return None
    return resp.json()


def all_details() -> list[dict[str, Any]]:
    rows = call("GET", "/tickets") or []
    return [d for r in rows if (d := call("GET", f"/tickets/{r['id']}"))]


# --- views -----------------------------------------------------------------------
def sidebar() -> None:
    st.sidebar.header("Load sample ticket")
    samples = load_samples()
    name = st.sidebar.selectbox("Sample", list(samples))
    st.sidebar.caption(samples[name]["text"])
    if st.sidebar.button("Submit sample", type="primary"):
        created = call("POST", "/tickets", json=samples[name])
        if created:
            st.sidebar.success(f"Ticket #{created['id']}: {created['status']}")
    if st.sidebar.button("Refresh"):
        st.rerun()


def view_overview() -> None:
    rows = [
        {"id": t["id"], "customer": t["customer_id"], "status": t["status"], "category": t["category"] or "—",
         "urgency": t["urgency"] or "—", "sla": sla_label(t),
         "updated (UTC)": datetime.fromisoformat(t["updated_at"]).strftime("%Y-%m-%d %H:%M")}
        for t in all_details()
    ]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No tickets yet. Load a sample from the sidebar.")


def view_queue() -> None:
    queue = call("GET", "/queue") or []
    if not queue:
        st.info("Nothing awaiting human action.")
    for t in queue:
        with st.expander(f"#{t['id']} · {t['category']} · {t['urgency']} · SLA {sla_label(t)}", expanded=True):
            render_ticket_for_review(t)
            render_actions(t)


def render_ticket_for_review(t: dict[str, Any]) -> None:
    decision, draft = t["policy_decision"], t["draft"]
    st.markdown(f"**Ticket text**\n\n> {t['raw_text']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Extracted fields**")
        st.json({
            "customer_id": t["customer_id"], "category": t["category"], "urgency": t["urgency"],
            "sentiment": t["sentiment"], "confidence": t["confidence_score"],
            "refund": t["requested_refund_amount"],
        })
    with right:
        st.markdown("**Policy decision**")
        if decision:
            st.write("Triggered rules:", ", ".join(decision["triggered_rules"]) or "none")
            st.write("Allowed actions:", ", ".join(decision["allowed_actions"]))
            st.write("Required approver:", decision["required_approver"])
        if t["failsafe_reason"]:
            st.warning(f"Fail-safe: {t['failsafe_reason']}. An edited draft is required to approve.")
    if draft:
        st.markdown(f"**LLM reasoning:** {draft['reasoning']}")


def render_actions(t: dict[str, Any]) -> None:
    key = f"t{t['id']}"
    original = (t["draft"] or {}).get("draft_text", "")
    draft_text = st.text_area("Draft reply (editable)", original, key=f"{key}-draft", height=140)
    c1, c2, c3 = st.columns(3)
    reviewer = c1.text_input("Reviewer name", key=f"{key}-rev")
    role = c2.selectbox("Approver role", ROLES, key=f"{key}-role")
    reason = c3.text_input("Reason", key=f"{key}-reason")
    approve_col, reject_col = st.columns(2)

    with approve_col:
        if st.button("Approve", key=f"{key}-approve", type="primary"):
            body = {"reviewer": reviewer, "reason": reason, "approver_role": role}
            if draft_text.strip() != original.strip():
                body["edited_draft"] = draft_text
            submit_action(t["id"], "approve", body)

    with reject_col:
        action = st.radio("Reject as", ["override", "close"], horizontal=True, key=f"{key}-act")
        override: dict[str, Any] = {}
        if action == "override":
            for field, options in (("category", CATEGORIES), ("urgency", URGENCIES), ("sentiment", SENTIMENTS)):
                choice = st.selectbox(f"Override {field}", [UNCHANGED, *options], key=f"{key}-{field}")
                if choice != UNCHANGED:
                    override[field] = choice
            refund = st.text_input("Override refund amount (USD)", key=f"{key}-refund")
            if refund.strip():
                override["requested_refund_amount"] = refund.strip()
        if st.button("Reject", key=f"{key}-reject"):
            body = {"reviewer": reviewer, "reason": reason, "approver_role": role, "action": action}
            if action == "override":
                body["override"] = override
            submit_action(t["id"], "reject", body)


def submit_action(ticket_id: int, verb: str, body: dict[str, Any]) -> None:
    result = call("POST", f"/tickets/{ticket_id}/{verb}", json=body)
    if result:
        st.session_state.flash = f"Ticket #{ticket_id}: {verb} done, now {result['status']}"
        st.rerun()


def view_timeline() -> None:
    tickets = call("GET", "/tickets") or []
    if not tickets:
        st.info("No tickets yet.")
        return
    ticket_id = st.selectbox("Ticket", [t["id"] for t in tickets], format_func=lambda i: f"#{i}")
    for e in call("GET", f"/tickets/{ticket_id}/audit") or []:
        st.markdown(f"`{e['timestamp']}` **{e['event_type']}** · {e['actor']}  \n{audit_summary(e)}")


def view_audit() -> None:
    status = call("GET", "/audit/verify")
    if status:
        if status["valid"]:
            st.success(f"Chain valid: {status['checked']} entries verified")
        else:
            st.error(f"CHAIN BROKEN at entry id {status['first_bad_id']} ({status['checked']} checked)")
    ids = [t["id"] for t in call("GET", "/tickets") or []]
    ticket_id = st.selectbox("Ticket", ids, format_func=lambda i: f"#{i}", key="audit-ticket") if ids else None
    if ticket_id is not None:
        entries = call("GET", f"/tickets/{ticket_id}/audit") or []
        st.dataframe(
            [{"id": e["id"], "time": e["timestamp"], "event": e["event_type"], "actor": e["actor"],
              "entry_hash": e["entry_hash"][:12], "prev_hash": e["prev_hash"][:12]} for e in entries],
            hide_index=True, use_container_width=True,
        )
        with st.expander("Full payloads"):
            st.json([{"id": e["id"], "payload": e["payload"]} for e in entries])


def main() -> None:
    st.set_page_config(page_title="Intake-to-Resolution", layout="wide")
    st.title("Intake-to-Resolution")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    sidebar()
    tabs = st.tabs(["Tickets", "Approval queue", "Timeline", "Audit log"])
    for tab, view in zip(tabs, (view_overview, view_queue, view_timeline, view_audit)):
        with tab:
            view()


if __name__ == "__main__":
    main()

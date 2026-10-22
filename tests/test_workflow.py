import json

import pytest
from sqlmodel import Session, select

from database.models import StateSnapshot
from engine.policy_engine import can_approve, enforce_action
from engine.schemas import (
    ActionType, Approver, ClassificationOverride, HumanAction, HumanDecision, PolicyDecision,
)
from engine.workflow import is_paused, resume_ticket
from tests.conftest import CREATED_AT, load_fixture, make_env


def human(decision=HumanAction.APPROVE, **kw) -> HumanDecision:
    base = dict(reviewer="alice", approver_role=Approver.MANAGER)
    if decision is not HumanAction.APPROVE:
        base["reason"] = "needs rework"
    return HumanDecision(decision=decision, **{**base, **kw})


def payloads(env, tid, event):
    return [json.loads(r.payload_json) for r in env.audit(tid) if r.event_type == event]


# --- Auto-resolve -------------------------------------------------------------
def test_auto_resolve_happy_path(env):
    tid = env.run("known_resolution_technical")
    assert env.status(tid) == "RESOLVED"
    assert not is_paused(env.graph, tid)
    assert env.events(tid) == [
        "TICKET_CREATED", "STATE_TRANSITION", "LLM_CALL", "POLICY_CHECK", "LLM_CALL", "DECISION", "STATE_TRANSITION",
    ]
    assert payloads(env, tid, "DECISION")[0]["final_action"] == "AUTO_RESOLVE"
    env.assert_chain_valid()


def test_llm_calls_are_audited_with_hashes(env):
    tid = env.run("known_resolution_technical")
    calls = payloads(env, tid, "LLM_CALL")
    assert [c["task"] for c in calls] == ["extract", "draft"]
    assert all(c["provider"] == "mock" and c["model"] and len(c["prompt_hash"]) == 64 and c["outcome"] == "validated" for c in calls)


# --- Pause triggers -----------------------------------------------------------
@pytest.mark.parametrize(
    "fixture,rule,overridden",
    [
        ("billing_refund_over_250", "RULE_2_REFUND_APPROVAL", False),
        ("vip_frustrated_technical", "RULE_1_VIP_ESCALATION", True),
        ("low_confidence_vague", "LOW_CONFIDENCE", True),
        ("legal_terms", "LEGAL_TERMS", True),
    ],
)
def test_pause_triggers_await_human(env, fixture, rule, overridden):
    tid = env.run(fixture)
    assert env.status(tid) == "AWAITING_HUMAN_ACTION"
    assert is_paused(env.graph, tid)
    assert rule in payloads(env, tid, "POLICY_CHECK")[0]["decision"]["triggered_rules"]
    assert bool(payloads(env, tid, "ENGINE_OVERRIDE")) is overridden
    env.assert_chain_valid()


def test_engine_overrides_llm_auto_resolve_and_audits_it(env):
    tid = env.run("legal_terms")  # LLM proposes AUTO_RESOLVE, legal term forbids it
    o = payloads(env, tid, "ENGINE_OVERRIDE")[0]
    assert o["proposed"] == "AUTO_RESOLVE" and o["final"] == "REQUIRES_HUMAN_APPROVAL"
    assert o["allowed"] == ["REQUIRES_HUMAN_APPROVAL"] and o["triggered_rules"] == ["LEGAL_TERMS"]
    assert env.status(tid) != "RESOLVED"


def test_vip_tier_comes_from_customers_json(env):
    tid = env.run("vip_frustrated_technical")
    assert payloads(env, tid, "POLICY_CHECK")[0]["tier"] == "VIP"


def test_kb_gap_routes_to_human(tmp_path):
    kb_without_technical = "# KB\n\n## Billing\n- x\n"
    e = make_env(f"sqlite:///{tmp_path / 'a.db'}", tmp_path / "c.db", kb=kb_without_technical)
    tid = e.run("known_resolution_technical")  # would otherwise auto-resolve
    assert e.status(tid) == "AWAITING_HUMAN_ACTION"
    assert "KB_NOT_COVERED" in e.events(tid)
    state = e.graph.get_state({"configurable": {"thread_id": f"ticket-{tid}"}}).values
    assert "does not cover" in state["draft"]["draft_text"]
    # approving the "not covered" stub as-is is refused; a human must write the reply
    with pytest.raises(ValueError):
        resume_ticket(e.graph, tid, human())
    resume_ticket(e.graph, tid, human(edited_draft="Please restart the app."))
    assert e.status(tid) == "RESOLVED"
    e.assert_chain_valid()
    e.graph.checkpointer.conn.close()


# --- Fail-safe ----------------------------------------------------------------
def test_malformed_output_fails_safe_to_human(env):
    tid = env.run("malformed_output")
    assert env.status(tid) == "AWAITING_HUMAN_ACTION"
    calls = payloads(env, tid, "LLM_CALL")
    assert [c["outcome"] for c in calls] == ["failed", "failed"]  # tried once, retried once
    fs = payloads(env, tid, "FAIL_SAFE")[0]
    assert fs["stage"] == "extract" and fs["route"] == "REQUIRES_HUMAN_APPROVAL"
    assert payloads(env, tid, "DECISION")[0]["basis"] == "extraction_failed"
    assert "POLICY_CHECK" not in env.events(tid)
    env.assert_chain_valid()


def test_retry_recovers_then_auto_resolves(env):
    tid = env.run("retry_recovers")
    calls = payloads(env, tid, "LLM_CALL")
    assert [(c["task"], c["outcome"]) for c in calls] == [("extract", "failed"), ("extract", "validated"), ("draft", "validated")]
    assert env.status(tid) == "RESOLVED"


def test_failsafe_ticket_can_be_rescued_by_override(env):
    tid = env.run("malformed_output")
    ov = ClassificationOverride(category="Technical", urgency="Low", sentiment="Neutral",
                                extracted_entities={"resolution_code": "RES-PWD-RESET"})
    resume_ticket(env.graph, tid, human(HumanAction.REJECT_OVERRIDE, override=ov))
    assert "POLICY_CHECK" in env.events(tid)  # re-ran policy_check
    assert env.status(tid) == "AWAITING_HUMAN_ACTION"  # no draft fixture, draft call fails -> still human
    assert "draft" in [p["stage"] for p in payloads(env, tid, "FAIL_SAFE")]


# --- Resume paths -------------------------------------------------------------
def test_approve_with_edited_draft_completes(env):
    tid = env.run("billing_refund_over_250")
    resume_ticket(env.graph, tid, human(edited_draft="Refund approved, 5-7 business days."))
    assert env.status(tid) == "RESOLVED"
    assert not is_paused(env.graph, tid)
    h = payloads(env, tid, "HUMAN_DECISION")[0]
    assert h["draft_edited"] and h["original_draft_hash"] != h["final_draft_hash"]
    assert [r.actor for r in env.audit(tid) if r.event_type == "HUMAN_DECISION"] == ["human:alice"]
    env.assert_chain_valid()


def test_approve_requires_sufficient_role(env):
    tid = env.run("billing_refund_over_250")  # needs MANAGER
    with pytest.raises(PermissionError):
        resume_ticket(env.graph, tid, human(approver_role=Approver.SUPPORT_AGENT))
    assert env.status(tid) == "AWAITING_HUMAN_ACTION" and is_paused(env.graph, tid)
    assert "HUMAN_DECISION" not in env.events(tid)


def test_reject_override_reexecutes_from_policy_check(env):
    tid = env.run("low_confidence_vague")
    assert env.status(tid) == "AWAITING_HUMAN_ACTION"
    ov = ClassificationOverride(category="Technical", extracted_entities={"resolution_code": "RES-CLEAR-CACHE"})
    resume_ticket(env.graph, tid, human(HumanAction.REJECT_OVERRIDE, override=ov))
    # human vouches (confidence 1.0) + known code -> rule 3 -> LLM's AUTO_RESOLVE now allowed
    assert env.status(tid) == "RESOLVED"
    checks = payloads(env, tid, "POLICY_CHECK")
    assert len(checks) == 2
    assert "LOW_CONFIDENCE" in checks[0]["decision"]["triggered_rules"]
    assert checks[1]["decision"]["triggered_rules"] == ["RULE_3_AUTO_RESOLVE"]
    transitions = [(p["from"], p["to"]) for p in payloads(env, tid, "STATE_TRANSITION")]
    assert transitions[-3:] == [("AWAITING_HUMAN_ACTION", "REJECTED"), ("REJECTED", "PROCESSING"),
                                ("PROCESSING", "RESOLVED")]
    env.assert_chain_valid()


def test_override_cannot_bypass_engine(env):
    tid = env.run("low_confidence_vague")
    ov = ClassificationOverride(category="Billing", requested_refund_amount="999.00")
    resume_ticket(env.graph, tid, human(HumanAction.REJECT_OVERRIDE, override=ov))
    assert env.status(tid) == "AWAITING_HUMAN_ACTION"  # rule 2 now requires manager approval
    assert "RULE_2_REFUND_APPROVAL" in payloads(env, tid, "POLICY_CHECK")[-1]["decision"]["triggered_rules"]


def test_reject_close_ends_workflow(env):
    tid = env.run("billing_refund_over_250")
    resume_ticket(env.graph, tid, human(HumanAction.REJECT_CLOSE, reason="duplicate of #12"))
    assert env.status(tid) == "CLOSED"
    assert not is_paused(env.graph, tid)
    assert [(p["from"], p["to"]) for p in payloads(env, tid, "STATE_TRANSITION")][-2:] == [
        ("AWAITING_HUMAN_ACTION", "REJECTED"), ("REJECTED", "CLOSED")]
    with pytest.raises(ValueError):
        resume_ticket(env.graph, tid, human())
    env.assert_chain_valid()


def test_invalid_override_rejected_before_resume(env):
    tid = env.run("malformed_output")  # nothing extracted, so override must supply a full classification
    with pytest.raises(Exception):
        resume_ticket(env.graph, tid, human(HumanAction.REJECT_OVERRIDE, override=ClassificationOverride(category="Billing")))
    assert is_paused(env.graph, tid)


# --- Restart ------------------------------------------------------------------
def test_resume_after_simulated_restart(env):
    tid = env.run("billing_refund_over_250")
    env2 = env.restart()
    try:
        assert env2.graph is not env.graph
        assert is_paused(env2.graph, tid)
        resume_ticket(env2.graph, tid, human(edited_draft="Approved after restart."))
        assert env2.status(tid) == "RESOLVED"
        env2.assert_chain_valid()
    finally:
        env2.graph.checkpointer.conn.close()


def test_override_loop_survives_restart(env):
    tid = env.run("low_confidence_vague")
    env2 = env.restart()
    try:
        ov = ClassificationOverride(category="Technical", extracted_entities={"resolution_code": "RES-CLEAR-CACHE"})
        resume_ticket(env2.graph, tid, human(HumanAction.REJECT_OVERRIDE, override=ov))
        assert env2.status(tid) == "RESOLVED"
    finally:
        env2.graph.checkpointer.conn.close()


# --- Audit completeness -------------------------------------------------------
@pytest.mark.parametrize("fixture", [
    "known_resolution_technical", "billing_refund_over_250", "vip_frustrated_technical",
    "low_confidence_vague", "legal_terms", "malformed_output", "retry_recovers",
])
def test_every_scenario_audited_and_chain_verifies(env, fixture):
    tid = env.run(fixture)
    ev = env.events(tid)
    assert ev[0] == "TICKET_CREATED" and "DECISION" in ev and "LLM_CALL" in ev
    # every status change the ticket made is in the log
    last_to = [p["to"] for p in payloads(env, tid, "STATE_TRANSITION")][-1]
    assert last_to == env.status(tid)
    env.assert_chain_valid()


def test_snapshots_written(env):
    tid = env.run("billing_refund_over_250")
    with Session(env.deps.engine) as s:
        steps = [r.step for r in s.exec(select(StateSnapshot).where(StateSnapshot.ticket_id == tid))]
    assert steps == ["decide"]


# --- Pure helpers -------------------------------------------------------------
def decision(allowed):
    return PolicyDecision(triggered_rules=[], allowed_actions=allowed, required_approver=Approver.NONE,
                          sla_hours=8, sla_deadline=CREATED_AT)


def test_enforce_action():
    only_human = decision([ActionType.REQUIRES_HUMAN_APPROVAL])
    assert enforce_action(ActionType.AUTO_RESOLVE, only_human) == (ActionType.REQUIRES_HUMAN_APPROVAL, True)
    assert enforce_action(ActionType.REASSIGN_TIER_2, only_human) == (ActionType.REQUIRES_HUMAN_APPROVAL, True)
    full = decision(list(ActionType))
    assert enforce_action(ActionType.AUTO_RESOLVE, full) == (ActionType.AUTO_RESOLVE, False)


def test_can_approve():
    assert can_approve(Approver.MANAGER, Approver.SUPPORT_AGENT)
    assert not can_approve(Approver.SUPPORT_AGENT, Approver.MANAGER)


@pytest.mark.parametrize("kw", [
    dict(decision="APPROVE", override={"category": "Billing"}),
    dict(decision="REJECT_OVERRIDE", reason="x"),
    dict(decision="REJECT_OVERRIDE", reason="x", override={}),
    dict(decision="REJECT_CLOSE"),
    dict(decision="REJECT_CLOSE", reason="x", edited_draft="y"),
    dict(decision="APPROVE", approver_role="NONE"),
])
def test_human_decision_shape_validation(kw):
    with pytest.raises(Exception):
        HumanDecision(reviewer="a", **kw)

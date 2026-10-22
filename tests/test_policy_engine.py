from datetime import timedelta

import pytest

from engine.policy_engine import contains_legal_terms, evaluate_policy, lookup_tier
from engine.schemas import ActionType, Approver, CustomerTier, TriggeredRule as R
from tests.conftest import CREATED_AT

STD, VIP = CustomerTier.STANDARD, CustomerTier.VIP


@pytest.fixture
def decide(rules, known_codes):
    def _decide(ticket, tier=STD, text=""):
        return evaluate_policy(ticket, tier, rules, known_codes, CREATED_AT, text)

    return _decide


# --- Rule 1: VIP escalation -------------------------------------------------
@pytest.mark.parametrize("urgency", ["High", "Critical"])
def test_rule1_vip_high_urgency(decide, make_ticket, urgency):
    d = decide(make_ticket(urgency=urgency), VIP)
    assert R.RULE_1_VIP_ESCALATION in d.triggered_rules


def test_rule1_vip_frustrated(decide, make_ticket):
    assert R.RULE_1_VIP_ESCALATION in decide(make_ticket(sentiment="Frustrated"), VIP).triggered_rules


@pytest.mark.parametrize("urgency", ["Medium", "Low"])
def test_rule1_not_fired_vip_calm(decide, make_ticket, urgency):
    d = decide(make_ticket(urgency=urgency), VIP)
    assert R.RULE_1_VIP_ESCALATION not in d.triggered_rules


def test_rule1_not_fired_for_standard(decide, make_ticket):
    d = decide(make_ticket(urgency="Critical", sentiment="Frustrated"), STD)
    assert R.RULE_1_VIP_ESCALATION not in d.triggered_rules


# --- Rule 2: refund threshold ----------------------------------------------
@pytest.mark.parametrize(
    "amount,fires", [("250.00", False), ("250.01", True), ("249.99", False), ("1000", True), (None, False)]
)
def test_rule2_boundary(decide, make_ticket, amount, fires):
    d = decide(make_ticket(category="Billing", requested_refund_amount=amount))
    assert (R.RULE_2_REFUND_APPROVAL in d.triggered_rules) is fires
    if fires:
        assert d.required_approver is Approver.MANAGER
        assert d.allowed_actions == [ActionType.REQUIRES_HUMAN_APPROVAL]


def test_rule2_only_for_billing(decide, make_ticket):
    d = decide(make_ticket(category="Account", requested_refund_amount="999"))
    assert R.RULE_2_REFUND_APPROVAL not in d.triggered_rules


# --- Rule 3: auto-resolve ---------------------------------------------------
def technical(make_ticket, **kw):
    return make_ticket(category="Technical", extracted_entities={"resolution_code": "RES-PWD-RESET"}, **kw)


def test_rule3_fires_and_allows_auto_resolve(decide, make_ticket):
    d = decide(technical(make_ticket))
    assert d.triggered_rules == [R.RULE_3_AUTO_RESOLVE]
    assert ActionType.AUTO_RESOLVE in d.allowed_actions
    assert d.required_approver is Approver.NONE


@pytest.mark.parametrize("score,fires", [(0.85, True), (0.849, False), (0.9, True), (0.0, False)])
def test_confidence_boundary(decide, make_ticket, score, fires):
    d = decide(technical(make_ticket, confidence_score=score))
    assert (R.RULE_3_AUTO_RESOLVE in d.triggered_rules) is fires
    assert (R.LOW_CONFIDENCE in d.triggered_rules) is (not fires)
    assert (ActionType.AUTO_RESOLVE in d.allowed_actions) is fires


def test_rule3_needs_known_code(decide, make_ticket):
    for entities in ({}, {"resolution_code": "RES-MADE-UP"}):
        d = decide(make_ticket(category="Technical", extracted_entities=entities))
        assert ActionType.AUTO_RESOLVE not in d.allowed_actions


def test_rule3_needs_technical(decide, make_ticket):
    t = make_ticket(category="Account", extracted_entities={"resolution_code": "RES-PWD-RESET"})
    assert ActionType.AUTO_RESOLVE not in decide(t).allowed_actions


# --- Independent triggers ---------------------------------------------------
@pytest.mark.parametrize(
    "text", ["I will file a lawsuit", "my ATTORNEY called", "considering legal action", "I'll sue you", "the regulator"]
)
def test_legal_terms_force_human(decide, make_ticket, text):
    d = decide(technical(make_ticket), text=text)
    assert R.LEGAL_TERMS in d.triggered_rules
    assert d.allowed_actions == [ActionType.REQUIRES_HUMAN_APPROVAL]


def test_legal_keyword_is_whole_word(rules):
    assert not contains_legal_terms("this issue is unresolved, pursue it", rules.legal_keywords)


def test_vip_alone_blocks_auto_resolve(decide, make_ticket):
    d = decide(technical(make_ticket), VIP)
    assert d.triggered_rules == [R.VIP_TIER]
    assert d.allowed_actions == [ActionType.REQUIRES_HUMAN_APPROVAL]
    assert d.required_approver is Approver.SUPPORT_AGENT


def test_no_rule_general_ticket_cannot_auto_resolve(decide, make_ticket):
    d = decide(make_ticket())
    assert d.triggered_rules == []
    assert ActionType.AUTO_RESOLVE not in d.allowed_actions
    assert ActionType.REASSIGN_TIER_2 in d.allowed_actions


# --- Precedence -------------------------------------------------------------
def test_rule1_outranks_rule2_and_rule3(decide, make_ticket):
    t = make_ticket(
        category="Billing", urgency="Critical", requested_refund_amount="500",
        extracted_entities={"resolution_code": "RES-PWD-RESET"},
    )
    d = decide(t, VIP, text="legal action")
    assert d.triggered_rules[:2] == [R.RULE_1_VIP_ESCALATION, R.RULE_2_REFUND_APPROVAL]
    assert R.RULE_3_AUTO_RESOLVE not in d.triggered_rules
    assert d.allowed_actions == [ActionType.REQUIRES_HUMAN_APPROVAL]
    assert d.required_approver is Approver.MANAGER


def test_rule2_beats_rule3(decide, make_ticket):
    t = make_ticket(category="Billing", requested_refund_amount="300",
                    extracted_entities={"resolution_code": "RES-PWD-RESET"})
    d = decide(t)
    assert d.triggered_rules == [R.RULE_2_REFUND_APPROVAL]


def test_low_confidence_and_legal_together(decide, make_ticket):
    d = decide(technical(make_ticket, confidence_score=0.5), text="attorney")
    assert d.triggered_rules == [R.LEGAL_TERMS, R.LOW_CONFIDENCE]


# --- SLA --------------------------------------------------------------------
@pytest.mark.parametrize("urgency,hours", [("Critical", 2), ("High", 4), ("Medium", 8), ("Low", 24)])
def test_sla_deadline(decide, make_ticket, urgency, hours):
    d = decide(make_ticket(urgency=urgency))
    assert d.sla_hours == hours
    assert d.sla_deadline == CREATED_AT + timedelta(hours=hours)


def test_naive_created_at_rejected(rules, known_codes, make_ticket):
    from datetime import datetime
    with pytest.raises(ValueError):
        evaluate_policy(make_ticket(), STD, rules, known_codes, datetime(2026, 1, 1))


# --- Tier lookup ------------------------------------------------------------
def test_tier_lookup(customers):
    assert lookup_tier(customers, "C-1001") is VIP
    assert lookup_tier(customers, "C-2001") is STD
    assert lookup_tier(customers, "NOPE-999") is STD


def test_malformed_tier_defaults_standard():
    assert lookup_tier({"X": {"tier": "PLATINUM"}}, "X") is STD

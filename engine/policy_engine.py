"""Pure, deterministic policy evaluation. No I/O, no clock, no LLM."""
from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from datetime import datetime, timedelta

from engine.schemas import (
    ActionType,
    Approver,
    Category,
    CustomerTier,
    PolicyDecision,
    PolicyRules,
    TicketSchema,
    TriggeredRule,
)

RESOLUTION_CODE_ENTITY = "resolution_code"


def lookup_tier(customers: Mapping[str, Mapping[str, str]], customer_id: str) -> CustomerTier:
    """Deterministic tier lookup. Unknown or malformed entry means STANDARD."""
    entry = customers.get(customer_id)
    if entry is None:
        return CustomerTier.STANDARD
    try:
        return CustomerTier(entry.get("tier", ""))
    except ValueError:
        return CustomerTier.STANDARD


def compute_sla_deadline(rules: PolicyRules, ticket: TicketSchema, created_at: datetime) -> datetime:
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    return created_at + timedelta(hours=rules.sla_hours[ticket.urgency])


def contains_legal_terms(text: str, keywords: Collection[str]) -> bool:
    """Whole-word, case-insensitive match so 'sue' does not fire on 'issue'."""
    return any(
        re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text, re.IGNORECASE) for kw in keywords
    )


def _vip_escalation(ticket: TicketSchema, tier: CustomerTier, rules: PolicyRules) -> bool:
    return tier is CustomerTier.VIP and (
        ticket.urgency in rules.vip_escalation_urgencies
        or ticket.sentiment in rules.vip_escalation_sentiments
    )


def _refund_over_threshold(ticket: TicketSchema, rules: PolicyRules) -> bool:
    amount = ticket.requested_refund_amount
    return (
        ticket.category is Category.BILLING
        and amount is not None
        and amount > rules.refund_manager_approval_threshold  # strictly greater
    )


def _has_known_resolution(ticket: TicketSchema, known_codes: Collection[str]) -> bool:
    return ticket.extracted_entities.get(RESOLUTION_CODE_ENTITY) in known_codes


def _required_approver(triggered: list[TriggeredRule], human_needed: bool) -> Approver:
    if TriggeredRule.RULE_1_VIP_ESCALATION in triggered or TriggeredRule.RULE_2_REFUND_APPROVAL in triggered:
        return Approver.MANAGER
    return Approver.SUPPORT_AGENT if human_needed else Approver.NONE


def evaluate_policy(
    ticket: TicketSchema,
    tier: CustomerTier,
    rules: PolicyRules,
    known_resolution_codes: Collection[str],
    created_at: datetime,
    ticket_text: str = "",
) -> PolicyDecision:
    """Apply rules in precedence order. Triggered rules are listed highest precedence first."""
    triggered: list[TriggeredRule] = []
    if _vip_escalation(ticket, tier, rules):
        triggered.append(TriggeredRule.RULE_1_VIP_ESCALATION)
    if _refund_over_threshold(ticket, rules):
        triggered.append(TriggeredRule.RULE_2_REFUND_APPROVAL)
    if contains_legal_terms(ticket_text, rules.legal_keywords):
        triggered.append(TriggeredRule.LEGAL_TERMS)
    if ticket.confidence_score < rules.confidence_threshold:
        triggered.append(TriggeredRule.LOW_CONFIDENCE)
    if tier is CustomerTier.VIP:
        triggered.append(TriggeredRule.VIP_TIER)

    # Rule 3 only if nothing above fired (confidence is covered by LOW_CONFIDENCE).
    auto_candidate = (
        not triggered
        and ticket.category is Category.TECHNICAL
        and _has_known_resolution(ticket, known_resolution_codes)
    )
    if auto_candidate:
        triggered.append(TriggeredRule.RULE_3_AUTO_RESOLVE)

    human_forced = bool(triggered) and not auto_candidate
    if auto_candidate:
        allowed = [ActionType.AUTO_RESOLVE, ActionType.REQUIRES_HUMAN_APPROVAL, ActionType.REASSIGN_TIER_2]
    elif human_forced:
        allowed = [ActionType.REQUIRES_HUMAN_APPROVAL]
    else:
        allowed = [ActionType.REQUIRES_HUMAN_APPROVAL, ActionType.REASSIGN_TIER_2]

    return PolicyDecision(
        triggered_rules=triggered,
        allowed_actions=allowed,
        required_approver=_required_approver(triggered, human_needed=not auto_candidate),
        sla_hours=rules.sla_hours[ticket.urgency],
        sla_deadline=compute_sla_deadline(rules, ticket, created_at),
    )


_APPROVER_RANK = {Approver.NONE: 0, Approver.SUPPORT_AGENT: 1, Approver.MANAGER: 2}


def enforce_action(proposed: ActionType, decision: PolicyDecision) -> tuple[ActionType, bool]:
    """Engine wins: a proposal outside allowed_actions becomes REQUIRES_HUMAN_APPROVAL.

    Returns (final_action, overridden).
    """
    if proposed in decision.allowed_actions:
        return proposed, False
    return ActionType.REQUIRES_HUMAN_APPROVAL, True


def can_approve(role: Approver, required: Approver) -> bool:
    return _APPROVER_RANK[role] >= _APPROVER_RANK[required]

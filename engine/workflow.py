"""LangGraph workflow: ingest -> extract -> policy_check -> generate_response -> decide -> auto_resolve | human_wait.

Pause/resume uses langgraph.types.interrupt / Command(resume=...) with a SQLite checkpointer.
Nodes before an interrupt re-run on resume, so human_wait does no writes before interrupt().
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.engine import Engine
from sqlmodel import Session

from database.db import append_audit
from database.models import StateSnapshot, Ticket, utcnow
from engine.llm import (
    TASK_DRAFT,
    TASK_EXTRACT,
    LLMOutputError,
    LLMProvider,
    build_draft_prompt,
    build_extraction_prompt,
    kb_section,
    structured_call,
)
from engine.policy_engine import can_approve, enforce_action, evaluate_policy, lookup_tier
from engine.schemas import (
    ActionType,
    Approver,
    ClassificationOverride,
    DraftResponseAction,
    HumanAction,
    HumanDecision,
    PolicyDecision,
    PolicyRules,
    TicketSchema,
    TicketStatus as S,
)

SYSTEM = "system"
ENGINE = "policy_engine"

_ALLOWED_TRANSITIONS: dict[S, set[S]] = {
    S.NEW: {S.PROCESSING},
    S.PROCESSING: {S.AWAITING_HUMAN_ACTION, S.RESOLVED},
    S.AWAITING_HUMAN_ACTION: {S.RESOLVED, S.REJECTED},
    S.REJECTED: {S.PROCESSING, S.CLOSED},
    S.RESOLVED: set(),
    S.CLOSED: set(),
}


class InvalidTransition(Exception):
    pass


class WorkflowState(TypedDict, total=False):
    ticket_id: int
    customer_id: str
    extracted: dict[str, Any] | None
    decision: dict[str, Any] | None
    draft: dict[str, Any] | None
    final_action: str
    failsafe_reason: str | None
    human: dict[str, Any] | None


@dataclass
class WorkflowDeps:
    engine: Engine
    provider: LLMProvider
    rules: PolicyRules
    customers: dict[str, dict[str, str]]
    known_codes: frozenset[str]
    kb: str


# --- helpers ------------------------------------------------------------------
def _session(deps: WorkflowDeps) -> Session:
    return Session(deps.engine, expire_on_commit=False)


def _get_ticket(session: Session, ticket_id: int) -> Ticket:
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise LookupError(f"ticket {ticket_id} not found")
    return ticket


def as_utc(dt: datetime) -> datetime:
    # SQLite drops tzinfo; everything we store is UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def transition(session: Session, ticket: Ticket, new: S, actor: str, reason: str) -> None:
    if new not in _ALLOWED_TRANSITIONS[ticket.status]:
        raise InvalidTransition(f"{ticket.status.value} -> {new.value}")
    old = ticket.status
    ticket.status, ticket.updated_at = new, utcnow()
    session.add(ticket)
    append_audit(
        session,
        event_type="STATE_TRANSITION",
        actor=actor,
        ticket_id=ticket.id,
        payload={"from": old.value, "to": new.value, "reason": reason},
    )


def _snapshot(session: Session, ticket_id: int, step: str, state: WorkflowState) -> None:
    session.add(StateSnapshot(ticket_id=ticket_id, step=step, state_json=json.dumps(state, default=str)))


def _sync_ticket_fields(ticket: Ticket, extracted: TicketSchema) -> None:
    ticket.category = extracted.category.value
    ticket.urgency = extracted.urgency.value
    ticket.sentiment = extracted.sentiment.value
    ticket.confidence_score = extracted.confidence_score
    amount = extracted.requested_refund_amount
    ticket.requested_refund_amount = None if amount is None else str(amount)


def apply_override(extracted: dict[str, Any] | None, override: ClassificationOverride, customer_id: str) -> TicketSchema:
    """Merge a reviewer's override. The human vouches for the result, so confidence becomes 1.0."""
    merged: dict[str, Any] = {**(extracted or {}), "customer_id": customer_id, "confidence_score": 1.0}
    merged.update(override.model_dump(mode="json", exclude_unset=True))
    return TicketSchema.model_validate(merged)  # raises if e.g. category is still missing


def validate_human_decision(state: WorkflowState, human: HumanDecision) -> None:
    """Deterministic checks on a resume payload. Raises PermissionError/ValueError."""
    decision = PolicyDecision.model_validate(state["decision"]) if state.get("decision") else None
    required = decision.required_approver if decision else Approver.SUPPORT_AGENT
    if human.decision is HumanAction.APPROVE:
        if not can_approve(human.approver_role, required):
            raise PermissionError(f"{required.value} approval required, got {human.approver_role.value}")
        if state.get("failsafe_reason") and not human.edited_draft:
            raise ValueError("approving a fail-safe ticket requires an edited_draft")
    elif human.decision is HumanAction.REJECT_OVERRIDE:
        assert human.override is not None
        apply_override(state.get("extracted"), human.override, state["customer_id"])


# --- graph --------------------------------------------------------------------
def build_graph(deps: WorkflowDeps, checkpointer: SqliteSaver):  # type: ignore[no-untyped-def]
    def ingest(state: WorkflowState) -> WorkflowState:
        with _session(deps) as s:
            ticket = _get_ticket(s, state["ticket_id"])
            transition(s, ticket, S.PROCESSING, SYSTEM, "ingested")
            s.commit()
            return {"customer_id": ticket.customer_id, "failsafe_reason": None}

    def extract(state: WorkflowState) -> WorkflowState:
        with _session(deps) as s:
            ticket = _get_ticket(s, state["ticket_id"])
            try:
                parsed = structured_call(
                    deps.provider, s, task=TASK_EXTRACT, prompt=build_extraction_prompt(ticket.raw_text),
                    ticket_text=ticket.raw_text, schema=TicketSchema, ticket_id=ticket.id,
                )
            except LLMOutputError as exc:
                append_audit(s, event_type="FAIL_SAFE", actor=SYSTEM, ticket_id=ticket.id,
                             payload={"stage": "extract", "reason": str(exc)[:500], "route": ActionType.REQUIRES_HUMAN_APPROVAL.value})
                s.commit()
                return {"extracted": None, "failsafe_reason": "extraction_failed"}
            if parsed.customer_id != ticket.customer_id:
                append_audit(s, event_type="CUSTOMER_ID_MISMATCH", actor=SYSTEM, ticket_id=ticket.id,
                             payload={"llm": parsed.customer_id, "authoritative": ticket.customer_id})
            parsed = parsed.model_copy(update={"customer_id": ticket.customer_id})
            s.commit()
            return {"extracted": parsed.model_dump(mode="json")}

    def policy_check(state: WorkflowState) -> WorkflowState:
        extracted = TicketSchema.model_validate(state["extracted"])
        with _session(deps) as s:
            ticket = _get_ticket(s, state["ticket_id"])
            tier = lookup_tier(deps.customers, ticket.customer_id)  # never from LLM output
            decision = evaluate_policy(
                extracted, tier, deps.rules, deps.known_codes, as_utc(ticket.created_at), ticket.raw_text
            )
            _sync_ticket_fields(ticket, extracted)
            s.add(ticket)
            append_audit(s, event_type="POLICY_CHECK", actor=ENGINE, ticket_id=ticket.id,
                         payload={"tier": tier.value, "decision": decision.model_dump(mode="json")})
            s.commit()
            return {"decision": decision.model_dump(mode="json"), "draft": None,
                    "failsafe_reason": None, "human": None}

    def generate_response(state: WorkflowState) -> WorkflowState:
        extracted = TicketSchema.model_validate(state["extracted"])
        decision = PolicyDecision.model_validate(state["decision"])
        with _session(deps) as s:
            ticket = _get_ticket(s, state["ticket_id"])
            section = kb_section(deps.kb, extracted.category.value)
            if not section:
                append_audit(s, event_type="KB_NOT_COVERED", actor=SYSTEM, ticket_id=ticket.id,
                             payload={"category": extracted.category.value})
                s.commit()
                draft = DraftResponseAction(
                    action_type=ActionType.REQUIRES_HUMAN_APPROVAL,
                    reasoning=f"No knowledge-base section covers category {extracted.category.value}.",
                    draft_text="Our knowledge base does not cover this request. It has been routed to a human for review.",
                )
                return {"draft": draft.model_dump(mode="json"), "failsafe_reason": "kb_not_covered"}
            try:
                draft = structured_call(
                    deps.provider, s, task=TASK_DRAFT,
                    prompt=build_draft_prompt(ticket.raw_text, extracted, decision, section),
                    ticket_text=ticket.raw_text, schema=DraftResponseAction, ticket_id=ticket.id,
                )
            except LLMOutputError as exc:
                append_audit(s, event_type="FAIL_SAFE", actor=SYSTEM, ticket_id=ticket.id,
                             payload={"stage": "draft", "reason": str(exc)[:500], "route": ActionType.REQUIRES_HUMAN_APPROVAL.value})
                s.commit()
                return {"draft": None, "failsafe_reason": "draft_failed"}
            s.commit()
            return {"draft": draft.model_dump(mode="json")}

    def decide(state: WorkflowState) -> WorkflowState:
        human_only = ActionType.REQUIRES_HUMAN_APPROVAL
        with _session(deps) as s:
            ticket = _get_ticket(s, state["ticket_id"])
            draft = DraftResponseAction.model_validate(state["draft"]) if state.get("draft") else None
            reason = state.get("failsafe_reason")
            if reason or draft is None or not state.get("decision"):
                final, why = human_only, reason or "missing_input"  # never auto-resolve on an error
            else:
                decision = PolicyDecision.model_validate(state["decision"])
                final, overridden = enforce_action(draft.action_type, decision)
                why = "policy_override" if overridden else "llm_proposal_allowed"
                if overridden:
                    append_audit(s, event_type="ENGINE_OVERRIDE", actor=ENGINE, ticket_id=ticket.id,
                                 payload={"proposed": draft.action_type.value, "final": final.value,
                                          "allowed": [a.value for a in decision.allowed_actions],
                                          "triggered_rules": [r.value for r in decision.triggered_rules]})
            append_audit(s, event_type="DECISION", actor=ENGINE, ticket_id=ticket.id,
                         payload={"final_action": final.value, "basis": why,
                                  "proposed": draft.action_type.value if draft else None})
            if final is not ActionType.AUTO_RESOLVE:
                transition(s, ticket, S.AWAITING_HUMAN_ACTION, SYSTEM, f"paused: {why}")
            update: WorkflowState = {"final_action": final.value}
            _snapshot(s, ticket.id, "decide", {**state, **update})
            s.commit()
            return update

    def auto_resolve(state: WorkflowState) -> WorkflowState:
        with _session(deps) as s:
            ticket = _get_ticket(s, state["ticket_id"])
            transition(s, ticket, S.RESOLVED, SYSTEM, "auto-resolved")
            _snapshot(s, ticket.id, "auto_resolve", state)
            s.commit()
        return {}

    def human_wait(state: WorkflowState) -> WorkflowState:
        # Nothing before interrupt(): this node body re-runs from the top on resume.
        raw = interrupt({
            "ticket_id": state["ticket_id"],
            "final_action": state.get("final_action"),
            "failsafe_reason": state.get("failsafe_reason"),
            "decision": state.get("decision"),
            "draft": state.get("draft"),
        })
        human = HumanDecision.model_validate(raw)
        validate_human_decision(state, human)
        update: WorkflowState = {"human": human.model_dump(mode="json", exclude_unset=True)}
        with _session(deps) as s:
            ticket = _get_ticket(s, state["ticket_id"])
            original = (state.get("draft") or {}).get("draft_text", "")
            final_text = human.edited_draft or original
            append_audit(s, event_type="HUMAN_DECISION", actor=f"human:{human.reviewer}", ticket_id=ticket.id,
                         payload={"decision": human.decision.value, "role": human.approver_role.value,
                                  "reason": human.reason,
                                  "draft_edited": human.edited_draft is not None,
                                  "original_draft_hash": _sha(original), "final_draft_hash": _sha(final_text),
                                  "override": human.override.model_dump(mode="json", exclude_unset=True) if human.override else None})
            actor = f"human:{human.reviewer}"
            if human.decision is HumanAction.APPROVE:
                if human.edited_draft:
                    base = state.get("draft") or {
                        "action_type": ActionType.REQUIRES_HUMAN_APPROVAL.value,
                        "reasoning": f"Written by human reviewer {human.reviewer} (no valid LLM draft).",
                    }
                    update["draft"] = {**base, "draft_text": human.edited_draft}
                transition(s, ticket, S.RESOLVED, actor, "approved")
            else:
                transition(s, ticket, S.REJECTED, actor, human.reason or "")
                if human.decision is HumanAction.REJECT_OVERRIDE:
                    assert human.override is not None
                    new = apply_override(state.get("extracted"), human.override, ticket.customer_id)
                    update["extracted"] = new.model_dump(mode="json")
                    transition(s, ticket, S.PROCESSING, actor, "re-running from policy_check after override")
                else:
                    transition(s, ticket, S.CLOSED, actor, "manually closed")
            _snapshot(s, ticket.id, "human_wait", {**state, **update})
            s.commit()
        return update

    def after_extract(state: WorkflowState) -> str:
        return "policy_check" if state.get("extracted") else "decide"

    def after_decide(state: WorkflowState) -> str:
        return "auto_resolve" if state["final_action"] == ActionType.AUTO_RESOLVE.value else "human_wait"

    def after_human(state: WorkflowState) -> str:
        return "policy_check" if (state.get("human") or {}).get("decision") == HumanAction.REJECT_OVERRIDE.value else END

    g = StateGraph(WorkflowState)
    for name, fn in [("ingest", ingest), ("extract", extract), ("policy_check", policy_check),
                     ("generate_response", generate_response), ("decide", decide),
                     ("auto_resolve", auto_resolve), ("human_wait", human_wait)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "extract")
    g.add_conditional_edges("extract", after_extract, ["policy_check", "decide"])
    g.add_edge("policy_check", "generate_response")
    g.add_edge("generate_response", "decide")
    g.add_conditional_edges("decide", after_decide, ["auto_resolve", "human_wait"])
    g.add_conditional_edges("human_wait", after_human, ["policy_check", END])
    g.add_edge("auto_resolve", END)
    return g.compile(checkpointer=checkpointer)


# --- public API ---------------------------------------------------------------
def open_checkpointer(path: Path | str) -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def _config(ticket_id: int) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"ticket-{ticket_id}"}}


def create_ticket(deps: WorkflowDeps, customer_id: str, raw_text: str) -> int:
    with _session(deps) as s:
        ticket = Ticket(customer_id=customer_id, raw_text=raw_text)
        s.add(ticket)
        s.flush()
        append_audit(s, event_type="TICKET_CREATED", actor=SYSTEM, ticket_id=ticket.id,
                     payload={"customer_id": customer_id, "raw_text_hash": _sha(raw_text)})
        s.commit()
        return ticket.id  # type: ignore[return-value]


def start_ticket(graph, ticket_id: int) -> None:  # type: ignore[no-untyped-def]
    graph.invoke({"ticket_id": ticket_id}, _config(ticket_id))


def is_paused(graph, ticket_id: int) -> bool:  # type: ignore[no-untyped-def]
    return tuple(graph.get_state(_config(ticket_id)).next) == ("human_wait",)


def resume_ticket(graph, ticket_id: int, human: HumanDecision) -> None:  # type: ignore[no-untyped-def]
    """Validate the reviewer's input against the paused state, then resume from the checkpoint."""
    if not is_paused(graph, ticket_id):
        raise ValueError(f"ticket {ticket_id} is not awaiting human action")
    validate_human_decision(graph.get_state(_config(ticket_id)).values, human)
    graph.invoke(Command(resume=human.model_dump(mode="json", exclude_unset=True)), _config(ticket_id))

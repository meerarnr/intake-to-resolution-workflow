"""Strict Pydantic v2 schemas shared by the LLM layer, policy engine and DB."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Category(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    ACCOUNT = "Account"
    GENERAL = "General"


class Urgency(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    FRUSTRATED = "Frustrated"


class CustomerTier(str, Enum):
    STANDARD = "STANDARD"
    VIP = "VIP"


class ActionType(str, Enum):
    AUTO_RESOLVE = "AUTO_RESOLVE"
    REQUIRES_HUMAN_APPROVAL = "REQUIRES_HUMAN_APPROVAL"
    REASSIGN_TIER_2 = "REASSIGN_TIER_2"


class Approver(str, Enum):
    NONE = "NONE"
    SUPPORT_AGENT = "SUPPORT_AGENT"
    MANAGER = "MANAGER"


class TriggeredRule(str, Enum):
    RULE_1_VIP_ESCALATION = "RULE_1_VIP_ESCALATION"
    RULE_2_REFUND_APPROVAL = "RULE_2_REFUND_APPROVAL"
    RULE_3_AUTO_RESOLVE = "RULE_3_AUTO_RESOLVE"
    LEGAL_TERMS = "LEGAL_TERMS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    VIP_TIER = "VIP_TIER"


class TicketStatus(str, Enum):
    NEW = "NEW"
    PROCESSING = "PROCESSING"
    AWAITING_HUMAN_ACTION = "AWAITING_HUMAN_ACTION"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"


class TicketSchema(BaseModel):
    """Structured ticket as extracted by the LLM. Rejects unknown/out-of-range values."""

    model_config = ConfigDict(extra="forbid")

    customer_id: NonEmptyStr
    category: Category
    urgency: Urgency
    sentiment: Sentiment
    extracted_entities: dict[str, str] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    requested_refund_amount: Decimal | None = Field(
        default=None, ge=0, max_digits=12, decimal_places=2, allow_inf_nan=False
    )


class DraftResponseAction(BaseModel):
    """LLM-proposed action. The policy engine may override it."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    reasoning: NonEmptyStr
    draft_text: NonEmptyStr


class PolicyDecision(BaseModel):
    """Deterministic output of the policy engine."""

    model_config = ConfigDict(extra="forbid")

    triggered_rules: list[TriggeredRule]
    allowed_actions: list[ActionType] = Field(min_length=1)
    required_approver: Approver
    sla_hours: int = Field(gt=0)
    sla_deadline: datetime


class PolicyRules(BaseModel):
    """Typed view of data/policy_rules.json, the single source of truth."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(ge=0.0, le=1.0)
    refund_manager_approval_threshold: Decimal = Field(ge=0, allow_inf_nan=False)
    sla_hours: dict[Urgency, int]
    legal_keywords: list[NonEmptyStr]
    vip_escalation_urgencies: list[Urgency]
    vip_escalation_sentiments: list[Sentiment]

    @field_validator("sla_hours")
    @classmethod
    def _all_urgencies_positive(cls, v: dict[Urgency, int]) -> dict[Urgency, int]:
        missing = set(Urgency) - set(v)
        if missing:
            raise ValueError(f"sla_hours missing urgencies: {sorted(m.value for m in missing)}")
        if any(h <= 0 for h in v.values()):
            raise ValueError("sla_hours must be positive")
        return v


class HumanAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT_OVERRIDE = "REJECT_OVERRIDE"  # reject, reclassify, re-run from policy_check
    REJECT_CLOSE = "REJECT_CLOSE"  # reject and close manually


class ClassificationOverride(BaseModel):
    """Fields a reviewer may change. Unset fields keep the extracted value."""

    model_config = ConfigDict(extra="forbid")

    category: Category | None = None
    urgency: Urgency | None = None
    sentiment: Sentiment | None = None
    extracted_entities: dict[str, str] | None = None
    requested_refund_amount: Decimal | None = Field(
        default=None, ge=0, max_digits=12, decimal_places=2, allow_inf_nan=False
    )


class HumanDecision(BaseModel):
    """Resume payload for a paused ticket."""

    model_config = ConfigDict(extra="forbid")

    decision: HumanAction
    reviewer: NonEmptyStr
    approver_role: Approver = Approver.SUPPORT_AGENT
    reason: NonEmptyStr | None = None
    edited_draft: NonEmptyStr | None = None
    override: ClassificationOverride | None = None

    @model_validator(mode="after")
    def _shape_matches_decision(self) -> "HumanDecision":
        if self.approver_role is Approver.NONE:
            raise ValueError("approver_role must be a human role")
        if self.decision is HumanAction.APPROVE:
            if self.override is not None:
                raise ValueError("override is only valid with REJECT_OVERRIDE")
            return self
        if self.edited_draft is not None:
            raise ValueError("edited_draft is only valid with APPROVE")
        if self.reason is None:
            raise ValueError("reason is required when rejecting")
        if self.decision is HumanAction.REJECT_OVERRIDE:
            if self.override is None or not self.override.model_fields_set:
                raise ValueError("REJECT_OVERRIDE needs at least one override field")
        elif self.override is not None:
            raise ValueError("override is only valid with REJECT_OVERRIDE")
        return self

from decimal import Decimal

import pytest
from pydantic import ValidationError

from engine.schemas import DraftResponseAction


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_confidence_out_of_range_rejected(make_ticket, score):
    with pytest.raises(ValidationError):
        make_ticket(confidence_score=score)


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_confidence_bounds_accepted(make_ticket, score):
    assert make_ticket(confidence_score=score).confidence_score == score


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity", "10.001"])
def test_bad_refund_rejected(make_ticket, amount):
    with pytest.raises(ValidationError):
        make_ticket(requested_refund_amount=amount)


def test_refund_is_decimal(make_ticket):
    assert make_ticket(requested_refund_amount="250.01").requested_refund_amount == Decimal("250.01")


@pytest.mark.parametrize("field,value", [("category", "Sales"), ("urgency", "Urgent"), ("sentiment", "Angry")])
def test_bad_enum_rejected(make_ticket, field, value):
    with pytest.raises(ValidationError):
        make_ticket(**{field: value})


def test_blank_customer_and_extra_fields_rejected(make_ticket):
    with pytest.raises(ValidationError):
        make_ticket(customer_id="   ")
    with pytest.raises(ValidationError):
        make_ticket(vip=True)


def test_draft_action_rejects_unknown_action():
    with pytest.raises(ValidationError):
        DraftResponseAction(action_type="DELETE_ALL", reasoning="x", draft_text="y")

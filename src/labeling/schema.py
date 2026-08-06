from typing import Literal
from pydantic import BaseModel


Intent = Literal[
    "delay_disruption",
    "checkin_boarding_issue",
    "flight_cancellation_rebooking",
    "lost_luggage",
    "special_assistance",
    "general_complaint",
    "general_question",
    "praise_feedback",
    "spam_irrelevant",
    "other_unclear",
]

Urgency = Literal["high", "medium", "low"]

class TicketLabel(BaseModel):
    intent: Intent
    urgency: Urgency
    abusive: bool


if __name__ == "__main__":
    good = '{"intent": "lost_luggage", "urgency": "medium", "abusive": false}'
    label = TicketLabel.model_validate_json(good)
    print("OK:", label)
    print("intent μόνο του:", label.intent)

    bad = '{"intent": "baggage_problem", "urgency": "medium", "abusive": false}'
    label = TicketLabel.model_validate_json(bad)
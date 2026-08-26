from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError

from src.api.model import ADAPTER_ID, load_model, triage_text
from src.labeling.schema import Intent, TicketLabel, Urgency

ROUTES = {
    "delay_disruption": "operations_team",
    "checkin_boarding_issue": "airport_team",
    "flight_cancellation_rebooking": "rebooking_team",
    "lost_luggage": "baggage_team",
    "special_assistance": "special_services_team",
    "general_complaint": "customer_relations",
    "general_question": "customer_relations",
    "praise_feedback": "no_action",
    "spam_irrelevant": "discard",
    "other_unclear": "human_review",
}

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tokenizer"], state["model"] = load_model()
    yield
    state.clear()

app = FastAPI(title="Airline Support Ticket Triage", lifespan=lifespan)

class TriageRequest(BaseModel):
    text: str = Field(min_length=3, max_length=1000)

class TriageResponse(BaseModel):
    intent: Intent
    urgency: Urgency
    abusive: bool
    route_to: str
    low_confidence: bool
    model_version: str

@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    raw = triage_text(req.text, state["tokenizer"], state["model"])
    try:
        label = TicketLabel.model_validate_json(raw)
        low_confidence = False
    except ValidationError:
        label = TicketLabel(intent="other_unclear", urgency="low", abusive=False)
        low_confidence = True

    route = "human_escalation" if label.abusive else ROUTES[label.intent]

    return TriageResponse(
        intent=label.intent,
        urgency=label.urgency,
        abusive=label.abusive,
        route_to=route,
        low_confidence=low_confidence,
        model_version=ADAPTER_ID,
    )    

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}
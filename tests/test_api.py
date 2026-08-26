import pytest
from fastapi.testclient import TestClient

from src.api.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_triage_happy_path(client):
    r = client.post("/triage", json={"text": "my bags are lost and nobody is helping me get them back"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "lost_luggage"
    assert body["route_to"] == "baggage_team"
    assert body["low_confidence"] is False

def test_empty_text_rejected(client):
    r = client.post("/triage", json={"text": ""})
    assert r.status_code == 422


def test_too_long_text_rejected(client):
    r = client.post("/triage", json={"text": "x" * 2000})
    assert r.status_code == 422
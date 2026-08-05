from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "user_id": "test_user",
    "trip_start_date": "2026-10-01",
    "trip_end_date": "2026-10-10",
    "city": ["Abu Dhabi"],
    "daily_food_budget": "Medium",
    "daily_attraction_budget": "Medium",
    "cuisine_preferences": ["Italian"],
    "activity_preferences": ["Culture focused (Museum/Galleries/..)"],
    "attraction_environment": ["indoor"],
    "traveling_with_kids": False,
    "num_recommendations": 5,
}


def test_valid_request_works():
    r = client.post("/recommend-restaurants", json=VALID_PAYLOAD)
    assert r.status_code == 200


def test_end_date_before_start_date_rejected():
    bad = {**VALID_PAYLOAD, "trip_start_date": "2026-10-10", "trip_end_date": "2026-10-01"}
    r = client.post("/recommend-restaurants", json=bad)
    assert r.status_code == 422


def test_zero_recommendations_rejected():
    bad = {**VALID_PAYLOAD, "num_recommendations": 0}
    r = client.post("/recommend-restaurants", json=bad)
    assert r.status_code == 422


def test_too_many_recommendations_rejected():
    bad = {**VALID_PAYLOAD, "num_recommendations": 999}
    r = client.post("/recommend-restaurants", json=bad)
    assert r.status_code == 422


def test_overly_long_other_field_rejected():
    bad = {**VALID_PAYLOAD, "cuisine_other": "x" * 200}
    r = client.post("/recommend-restaurants", json=bad)
    assert r.status_code == 422


def test_blank_other_field_is_allowed():
    ok = {**VALID_PAYLOAD, "cuisine_other": "   "}
    r = client.post("/recommend-restaurants", json=ok)
    assert r.status_code == 200

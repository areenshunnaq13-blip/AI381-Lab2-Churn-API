
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "tenure_months": 24,
        "monthly_fee": 70.5,
        "contract": "one-year",
        "internet_service": "fiber",
        "support_calls": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "churn_probability" in response.json()

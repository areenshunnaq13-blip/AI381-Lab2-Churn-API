
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

def test_model_loads():
    from src.app.main import model

    assert model is not None
    assert hasattr(model, "predict_proba")


def test_predict_invalid_input():
    payload = {
        "tenure_months": 24,
        "monthly_fee": 70.5,
        "contract": "one-year",
        "internet_service": "fiber"
        # support_calls is missing
    }

    response = client.post("/predict", json=payload)

    assert response.status_code in (400, 422, 500)


def test_predict_edge_case():
    payload = {
        "tenure_months": 0,
        "monthly_fee": 0,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "support_calls": 0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    probability = response.json()["churn_probability"]
    assert 0 <= probability <= 1

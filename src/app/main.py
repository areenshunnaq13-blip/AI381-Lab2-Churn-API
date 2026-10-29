
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model.joblib"

FEATURES = [
    "tenure_months",
    "monthly_fee",
    "contract",
    "internet_service",
    "support_calls",
]

model = joblib.load(MODEL_PATH)

app = FastAPI(title="AI381 Lab 2 Churn API")


class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_fee: float = Field(ge=0)
    contract: str = Field(min_length=1)
    internet_service: str = Field(min_length=1)
    support_calls: int = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/predict")
def predict(customer: CustomerInput):
    import pandas as pd

    data = customer.model_dump()
    prediction_input = pd.DataFrame([data], columns=FEATURES)

    churn_probability = float(
        model.predict_proba(prediction_input)[0][1]
    )

    return {
        "churn_probability": churn_probability
    }

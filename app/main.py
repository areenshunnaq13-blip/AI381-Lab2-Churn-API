
import pandas as pd
import joblib
from fastapi import FastAPI

MODEL_PATH = "model.joblib"

FEATURES = [
    "tenure_months",
    "monthly_fee",
    "contract",
    "internet_service",
    "support_calls"
]

app = FastAPI(title="AI381 Lab 1 Churn API")

model = joblib.load(MODEL_PATH)
print("Model features:", model.feature_names_in_)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: dict):
    X = pd.DataFrame([customer])[FEATURES]
    p = float(model.predict_proba(X)[0, 1])
    return {"churn_probability": round(p, 3)}

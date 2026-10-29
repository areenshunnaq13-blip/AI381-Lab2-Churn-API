
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "churn_v2.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

TARGET = "churn"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_fee",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
]


def train():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[features].copy()
    y = df[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    train()

from pathlib import Path
import hashlib
import os
import time

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "churn_v2.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "churn-model-training"
REGISTERED_MODEL_NAME = "churn-model"

# Maximum allowed P95 latency for promotion
MAX_P95_LATENCY_MS = 100

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


def get_file_hash(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def calculate_p95_latency(model, X_test):
    latencies = []

    for _ in range(100):
        start_time = time.perf_counter()
        model.predict(X_test)
        end_time = time.perf_counter()

        latencies.append((end_time - start_time) * 1000)

    return float(pd.Series(latencies).quantile(0.95))


def calculate_slice_f1_scores(X_test, y_test, predictions):
    slice_scores = {}

    for contract_type in X_test["contract"].unique():
        mask = X_test["contract"] == contract_type

        if mask.sum() > 0:
            score = f1_score(
                y_test[mask],
                predictions[mask],
                zero_division=0
            )

            metric_name = (
                f"f1_contract_{contract_type.replace('-', '_')}"
            )
            slice_scores[metric_name] = score

    for service_type in X_test["internet_service"].unique():
        mask = X_test["internet_service"] == service_type

        if mask.sum() > 0:
            score = f1_score(
                y_test[mask],
                predictions[mask],
                zero_division=0
            )

            metric_name = (
                f"f1_internet_{service_type.replace('-', '_')}"
            )
            slice_scores[metric_name] = score

    return slice_scores


def train_model(
    model_name,
    classifier,
    X_train,
    X_test,
    y_train,
    y_test,
    data_hash
):
    preprocessor = create_preprocessor()

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("data_hash", data_hash)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        p95_latency = calculate_p95_latency(model, X_test)

        model_file = BASE_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_file)
        model_size = os.path.getsize(model_file)

        slice_scores = calculate_slice_f1_scores(
            X_test,
            y_test,
            predictions
        )

        # Log overall metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("p95_latency_ms", p95_latency)
        mlflow.log_metric("model_size_bytes", model_size)

        # Log data slice metrics
        for metric_name, score in slice_scores.items():
            mlflow.log_metric(metric_name, score)

        model_info = mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 score: {f1:.3f}")
        print(f"P95 latency: {p95_latency:.3f} ms")
        print(f"Model size: {model_size} bytes")

        print("\nF1 scores by data slice:")
        for metric_name, score in slice_scores.items():
            print(f"{metric_name}: {score:.3f}")

        return {
            "name": model_name,
            "model": model,
            "f1_score": f1,
            "accuracy": accuracy,
            "p95_latency_ms": p95_latency,
            "model_size_bytes": model_size,
            "slice_scores": slice_scores,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }


def register_champion(best_model):
    client = MlflowClient()

    print("\nChecking promotion gate...")

    # Register the candidate model
    model_version = mlflow.register_model(
        model_uri=best_model["model_uri"],
        name=REGISTERED_MODEL_NAME
    )

    candidate_version = model_version.version
    candidate_f1 = best_model["f1_score"]
    candidate_latency = best_model["p95_latency_ms"]

    try:
        # Get the current champion
        champion_version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            "champion"
        )

        champion_run = client.get_run(champion_version.run_id)

        champion_f1 = champion_run.data.metrics.get(
            "f1_score",
            0
        )

        print(
            f"\nCurrent champion version: "
            f"{champion_version.version}"
        )
        print(f"Current champion F1: {champion_f1:.3f}")
        print(f"Candidate version: {candidate_version}")
        print(f"Candidate F1: {candidate_f1:.3f}")
        print(
            f"Candidate P95 latency: "
            f"{candidate_latency:.3f} ms"
        )

        # Promotion conditions
        is_better = candidate_f1 > champion_f1
        is_fast_enough = (
            candidate_latency <= MAX_P95_LATENCY_MS
        )

        if is_better and is_fast_enough:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="champion",
                version=candidate_version
            )

            print("\nPROMOTION APPROVED 🏆")
            print(
                f"Champion updated to version "
                f"{candidate_version}"
            )

        else:
            print("\nPROMOTION REJECTED ❌")

            if not is_better:
                print(
                    "Candidate F1 is not better than "
                    "the current champion."
                )

            if not is_fast_enough:
                print(
                    f"Candidate latency exceeds the "
                    f"{MAX_P95_LATENCY_MS} ms budget."
                )

            print(
                "The current champion remains unchanged."
            )

    except mlflow.exceptions.RestException:
        # If no champion exists yet, promote the first model
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=candidate_version
        )

        print("\nNo existing champion found.")
        print("First candidate promoted as champion 🏆")

    print(f"\nRegistered model name: {REGISTERED_MODEL_NAME}")
    print(f"Registered version: {candidate_version}")


def train():
    df = pd.read_csv(DATA_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_hash = get_file_hash(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    missing = [
        col for col in features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    X = df[features].copy()
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidates = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000)
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                random_state=42
            )
        ),
    ]

    results = []

    for model_name, classifier in candidates:
        result = train_model(
            model_name,
            classifier,
            X_train,
            X_test,
            y_train,
            y_test,
            data_hash
        )
        results.append(result)

    # Select the best candidate based on F1 score
    best_model = max(
        results,
        key=lambda result: result["f1_score"]
    )

    # Save the best candidate locally
    joblib.dump(
        best_model["model"],
        MODEL_PATH
    )

    # Apply the promotion gate
    register_champion(best_model)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 50)

    for result in results:
        print(
            f"{result['name']}: "
            f"F1={result['f1_score']:.3f}, "
            f"P95={result['p95_latency_ms']:.3f} ms, "
            f"Size={result['model_size_bytes']} bytes"
        )

    print(
        "\nChampion candidate:",
        best_model["name"]
    )
    print(
        f"Best F1 score: "
        f"{best_model['f1_score']:.3f}"
    )
    print(
        f"Best candidate saved locally to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    train()
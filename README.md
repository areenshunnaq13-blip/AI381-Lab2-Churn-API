# AI381 Lab 2 & Lab 3 — Churn Prediction API and MLOps Pipeline

A containerized FastAPI service for customer churn prediction, with an automated MLOps pipeline for model training, evaluation, promotion, and Docker image publishing.

## Project Overview

This project develops a customer churn prediction API and extends it with an automated MLOps workflow.

The pipeline trains and evaluates three candidate models, tracks experiments using MLflow, applies a model promotion gate, and publishes a Docker image to GitHub Container Registry (GHCR).

## Features

- FastAPI REST API for customer churn prediction.
- Input validation and health check endpoint.
- Automated tests using pytest.
- Multi-stage Docker build.
- Non-root Docker runtime user.
- GitHub Actions CI workflow.
- Trivy vulnerability scanning.
- MLflow experiment tracking and model registry.
- Training and comparison of three candidate models.
- Model evaluation using overall F1 score and data slices.
- Automated champion model selection and promotion gate.
- MLflow model signature.
- Model lineage tracking using Git commit SHA and MLflow Run ID.
- Docker image publishing to GHCR.

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── train.yml
├── src/
│   ├── app/
│   │   └── main.py
│   └── train/
│       └── train.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── churn_v1.csv
├── churn_v2.csv
├── MODEL_SELECTION.md
└── README.md
```

## Requirements

- Python 3.11 or newer.
- Docker Desktop with Docker Compose.
- Git, if cloning the repository.

## Run the API

Clone the repository:

```bash
git clone https://github.com/areenshunnaq13-blip/AI381-Lab2-Churn-API.git
cd AI381-Lab2-Churn-API
```

Start the application:

```bash
docker compose up --build
```

The API will be available at:

- Health check: http://localhost:8000/health
- Interactive API documentation: http://localhost:8000/docs

Use `/docs` to inspect the prediction endpoint and submit a valid request to `/predict`.

To stop the application:

```bash
docker compose down
```

## Run Tests

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest
```

## MLOps Pipeline — Lab 3

### Model Training and Experiment Tracking

The training pipeline evaluates three candidate models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Each candidate is evaluated using:

- Accuracy
- F1 score
- P95 prediction latency
- Model size
- F1 score across customer data slices

Experiments, parameters, metrics, and model artifacts are tracked using MLflow.

The pipeline also records the dataset SHA256 hash and Git commit SHA to support reproducibility and traceability.

### Model Selection

Logistic Regression was selected as the champion model because it achieved the highest overall F1 score while maintaining a small model size and acceptable inference latency.

The detailed comparison and selection rationale are documented in `MODEL_SELECTION.md`.

### Model Registry and Promotion Gate

The selected model is registered in the MLflow Model Registry under:

```text
churn-model
```

The champion model is identified using the `champion` alias.

The promotion gate checks two conditions:

1. The candidate must achieve a higher F1 score than the current champion.
2. The candidate's P95 latency must remain below the configured limit of 100 ms.

If the candidate does not satisfy the promotion criteria, promotion is rejected and the existing champion remains unchanged.

The workflow also demonstrates a rejected promotion through a failed GitHub Actions run.

### Model Signature

The trained model is logged with an MLflow model signature describing its expected input and output schema.

This helps document and validate the model's prediction interface.

### Model Lineage

The pipeline generates a `lineage.json` artifact containing:

- Git commit SHA
- MLflow Run ID
- Registered model name
- Model version
- Selected model candidate
- Promotion status

The Docker image also includes metadata labels linking it to the corresponding Git commit and MLflow Run ID.

### MLflow State Persistence

The training workflow saves the MLflow database and model artifacts as GitHub Actions artifacts.

On subsequent runs, the previous MLflow state is restored before training, preserving experiment and model registry history across workflow executions.

## GitHub Actions and CI/CD

The repository uses two GitHub Actions workflows.

### CI Workflow — `ci.yml`

The CI workflow performs:

- Code linting using Ruff.
- Automated tests using pytest.
- Docker image building.
- Vulnerability scanning using Trivy.
- Docker image publishing to GHCR on pushes to `main`.

### Training Workflow — `train.yml`

The training workflow performs:

1. Install dependencies.
2. Restore previous MLflow state.
3. Start the MLflow tracking server.
4. Train and evaluate the three candidate models.
5. Apply the promotion gate.
6. Save the updated MLflow state.
7. Extract model lineage information.
8. Build the Docker image.
9. Scan the image using Trivy.
10. Publish the image to GHCR if promotion succeeds.

The image is tagged using the Git commit SHA.

## Docker Image — GitHub Container Registry

The Docker image is published to GitHub Container Registry.

**Image:**

```text
ghcr.io/areenshunnaq13-blip/ai381-lab2-churn-api
```

**Verified image tag:**

```text
8a458c6ab01bf166f7f02049deba0f9c94f632ba
```

**Image digest:**

```text
sha256:4c6d854e92ae3239e13c2bb28a79dedb5606f01668b852115d19105f30a60d79
```

**Pull command:**

```bash
docker pull ghcr.io/areenshunnaq13-blip/ai381-lab2-churn-api:8a458c6ab01bf166f7f02049deba0f9c94f632ba
```

The image metadata was verified using:

```bash
docker image inspect ghcr.io/areenshunnaq13-blip/ai381-lab2-churn-api:8a458c6ab01bf166f7f02049deba0f9c94f632ba --format '{{json .Config.Labels}}'
```

The image labels include:

- `org.opencontainers.image.revision`
- `ai381.mlflow.run_id`

These labels connect the published Docker image to its source code revision and MLflow training run.

## Repository

[AI381-Lab2-Churn-API on GitHub](https://github.com/areenshunnaq13-blip/AI381-Lab2-Churn-API)
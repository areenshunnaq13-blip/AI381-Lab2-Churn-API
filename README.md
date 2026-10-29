# AI381 Lab 1 · The Manual Pipeline Race (starter kit)

Dr. Tariq Al-Omari · AI381 Machine Learning Operations · JUST · Fall 2026/27

Lab 1 is short, a little chaotic, and on purpose. You will deliver a model by hand, change it, and
log what that costs you. There is no wrong result: an honest pain log is a great pain log.

## What is in this kit

| File | What it is |
|---|---|
| `churn_v1.csv` | Customer churn data, version 1 (1,200 customers) |
| `churn_v2.csv` | Version 2 of the same data. Something changed. You will find out in Twist 1 |
| `lab1_baseline.ipynb` | The notebook that trains a logistic regression and saves `model.joblib` |
| `app/main.py` | A small FastAPI service that loads `model.joblib` and serves `/predict` |
| `sample_request.json` | One customer to send to the API |
| `requirements.txt` | The packages the notebook author remembered to write down |
| `pain_log_template.txt` | Copy it to `pain_log.txt` and fill it in as you go |

## The race (start your stopwatch first)

1. **Round 1.** Run the notebook with `churn_v1.csv`. Start the API by hand:
   `uvicorn app.main:app --port 8000`
   Send three predictions with curl, changing the values each time:
   `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json`
2. **Twist 1.** Change the notebook to `churn_v2.csv`, retrain and redeploy. Find and fix what breaks.
3. **Twist 2.** Create a brand-new virtual environment (or a Codespace) and rebuild everything using only your own notes.
4. **Twist 3.** Answer: "Which model version is your API serving right now?" and explain why it is hard to prove.
5. Finish `pain_log.txt`: minutes per step, every failure, every unanswered question.

## Setup

- Python 3.11 or newer and a terminal. On Windows, use WSL2 with Ubuntu.
- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Turning it in

Record one 5–8 minute video with your face and voice, say each checkpoint aloud, and turn it in on the
**Lab 1** assignment in the **AI381 MLOPs** team on Microsoft Teams (Lab Submissions channel, Assignments tab).
The full instructions, checkpoints and rubric are on e-learning and on the Teams assignment.

Questions: Thursday online office hours on Teams, the Lab Submissions channel, or talomari@just.edu.jo.

# AI381 Lab 2 — Churn Prediction API

[![CI](https://github.com/areenshunnaq13-blip/AI381-Lab2-Churn-API/actions/workflows/ci.yml/badge.svg)](https://github.com/areenshunnaq13-blip/AI381-Lab2-Churn-API/actions/workflows/ci.yml)

A containerized FastAPI service for predicting customer churn probabilities using a trained machine learning model.

## Features

- FastAPI REST API.
- Customer churn prediction using a saved machine learning model.
- Input validation.
- Health check endpoint.
- Automated tests with pytest.
- Multi-stage Docker build.
- Automated GitHub Actions workflow.
- Trivy vulnerability scanning.
- Docker image publishing to GitHub Container Registry (GHCR), tagged by commit SHA.

## Requirements

- Docker Desktop with Docker Compose.
- Git, if cloning the repository.

## Run the service

Clone the repository and open its directory:

```bash
git clone https://github.com/areenshunnaq13-blip/AI381-Lab2-Churn-API.git
cd AI381-Lab2-Churn-API
```

Start the application with one command:

```bash
docker compose up --build
```

The API will be available at:

- Health check: http://localhost:8000/health
- Interactive API documentation: http://localhost:8000/docs

Use the documentation at `/docs` to inspect the prediction endpoint and submit a valid request to `/predict`.

To stop the service, press `Ctrl + C`. To remove the Compose container, run:

```bash
docker compose down
```

## Run tests

Install the dependencies and run pytest:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

## Docker and CI/CD

The Dockerfile uses a multi-stage build, runs the service as a non-root user, and defines a health check.

The GitHub Actions workflow runs tests, builds the Docker image, scans it with Trivy, and publishes the image to GHCR when the workflow succeeds on `main`. Published images are tagged with the commit SHA.

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
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
├── model.joblib
└── README.md
```

## Repository

https://github.com/areenshunnaq13-blip/AI381-Lab2-Churn-API

# Model Selection

## Candidate Models

Three candidate models were trained and compared:

| Model | F1 Score | P95 Latency | Model Size |
|---|---:|---:|---:|
| Logistic Regression | 0.587 | 11.218 ms | 4,490 bytes |
| Random Forest | 0.477 | 38.941 ms | 4,280,971 bytes |
| Gradient Boosting | 0.531 | 8.942 ms | 145,485 bytes |

## Data Slice Evaluation

The models were also evaluated across different data slices based on contract type and internet service.

The results showed performance differences across slices, particularly for the smaller groups such as customers with two-year contracts and no internet service.

## Champion Model

Logistic Regression was selected as the champion model because it achieved the highest overall F1 score of 0.587 among all candidate models.

In addition to having the best F1 score, Logistic Regression has a very small model size and acceptable prediction latency. Although Gradient Boosting achieved slightly lower latency, its F1 score was lower.

Therefore, Logistic Regression provides the best overall balance between predictive performance, model size, and latency for this churn prediction task.

**Selected Champion: Logistic Regression**
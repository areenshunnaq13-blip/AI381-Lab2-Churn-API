# Model Selection

## Candidate Models

Three candidate models were trained and evaluated using the same dataset and experimental setup:

| Model | F1 Score | P95 Latency | Model Size |
|---|---:|---:|---:|
| Logistic Regression | 0.587 | 33.0268 ms | 4,490 bytes |
| Random Forest | 0.477 | 84.4270 ms | 4,280,971 bytes |
| Gradient Boosting | 0.531 | 14.1069 ms | 145,485 bytes |

All three models achieved a P95 prediction latency below the configured limit of 100 ms.

## Data Slice Evaluation

The models were also evaluated across different data slices based on contract type and internet service.

### F1 Score by Data Slice

| Data Slice | Logistic Regression | Random Forest | Gradient Boosting |
|---|---:|---:|---:|
| Month-to-month contract | 0.643 | 0.503 | 0.575 |
| Two-year contract | 0.182 | 0.000 | 0.182 |
| One-year contract | 0.375 | 0.500 | 0.400 |
| No internet service | 0.111 | 0.160 | 0.091 |
| Fiber internet | 0.653 | 0.529 | 0.566 |
| DSL internet | 0.625 | 0.531 | 0.653 |

The results show that model performance varies across different customer groups. In particular, performance is lower for smaller groups, such as customers with two-year contracts and customers without internet service.

These slice-level results provide additional insight into model behavior beyond the overall F1 score.

## Champion Model Selection

Logistic Regression was selected as the champion model because it achieved the highest overall F1 score of 0.587 among the three candidates.

It also has the smallest model size at 4,490 bytes, making it significantly more lightweight than Random Forest and Gradient Boosting.

Although Gradient Boosting achieved lower prediction latency, its overall F1 score was lower than Logistic Regression.

Random Forest had the largest model size and the highest P95 latency among the three candidates.

Therefore, Logistic Regression provides the best overall balance between predictive performance, model size, and inference latency for this churn prediction task.

**Selected Champion: Logistic Regression**

## Promotion Criteria

The model promotion gate checks the following conditions:

1. The candidate model must achieve a higher F1 score than the current champion.
2. The candidate model's P95 latency must remain within the configured limit of 100 ms.

A candidate that does not satisfy the promotion criteria is rejected, and the existing champion remains unchanged.


# 💳 Credit Card Fraud Detection

[![CI](https://github.com/<your-username>/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/credit-card-fraud-detection/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end machine learning project that detects fraudulent card transactions in a highly imbalanced dataset (~0.3% fraud), from leakage-safe training and honest evaluation through to a tested, containerised REST API.

**What makes it more than a notebook:**

- Reusable, tested Python package (`src/fraud_detection`) instead of code that only lives in cells
- Models compared with **repeated cross-validation**, with SMOTE **inside** the pipeline so it never leaks into validation folds
- **Decision threshold chosen from out-of-fold training scores**, not left at 0.5
- Test results reported with **bootstrap confidence intervals**, because a handful of fraud cases makes point estimates unreliable
- One self-contained model file (pipeline + threshold + metadata), served by a validated **FastAPI** service
- Unit and API tests, linting, GitHub Actions CI and a Dockerfile

---

## Table of contents

- [Dataset](#dataset)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [API](#api)
- [Docker](#docker)
- [Development](#development)
- [Limitations](#limitations)
- [License](#license)

---

## Dataset

The project expects a CSV in the format of the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset:

| Column | Description |
| --- | --- |
| `Time` | Seconds since the first transaction in the file (**not used by the model**, see below) |
| `V1` – `V28` | Anonymised PCA components |
| `Amount` | Transaction amount |
| `Class` | Target: `0` legitimate, `1` fraud |

The sample used during development had 35,742 rows and 103 frauds (0.29%). After removing one unlabelled row and 140 duplicates, 35,601 rows remained. The data is **not** included in this repository: download `creditcard.csv` and place it in `data/`.

## Results

Running `make train` writes every number below to `reports/metrics.json` and the plots to `reports/figures/`. Copy the values from there once you have trained on your data.

| Model | Avg. precision (PR-AUC) | Precision | Recall |
| --- | --- | --- | --- |
| Earlier notebook: Random Forest + SMOTE, threshold 0.5, with `Time` | 0.779 | 0.85 | 0.81 |
| **This pipeline** (best of 5 candidates, tuned, tuned threshold) | _fill in from `metrics.json`_ | _…_ | _…_ |

Both rows use the same stratified test split (same seed), so they are directly comparable. The test set contains only about 21 fraud cases, so check the 95% confidence intervals in `metrics.json` before claiming that one model beats another. Differences smaller than the intervals are noise.

Figures produced by training: `precision_recall_curve.png`, `confusion_matrix.png`, `feature_importance.png` (permutation importance on the test set).

## Design decisions

| Decision | Why |
| --- | --- |
| **Primary metric: average precision** | With ~0.3% fraud, accuracy and ROC-AUC look excellent even for weak models. PR-based metrics reflect the cost of false alarms. |
| **SMOTE inside an `imblearn.Pipeline`** | Resampling happens only on the training part of each CV fold. Validation folds and the test set keep the real class balance. |
| **Class weights vs SMOTE both tested** | Which one works better is an empirical question, so both are candidates. |
| **Repeated stratified k-fold on the training set** | A single split with ~80 training frauds is too noisy to rank models. |
| **`Time` excluded** | It is "seconds since the first transaction in the file". A live transaction has no such value, and it would leave the training range within hours. |
| **`log1p` + `RobustScaler` on `Amount`** | `Amount` is heavily right-skewed. |
| **Threshold from out-of-fold scores, maximising F2** | A missed fraud usually costs more than a false alarm. F2 weights recall twice as much as precision. Adjust `THRESHOLD_BETA` in `config.py` to change the trade-off. |
| **Test set used once** | Selection, tuning and thresholding use training data only. |
| **API returns `fraud_score`, not "probability"** | Class weighting and SMOTE distort scores. They rank risk well but are not calibrated probabilities. |
| **Single model file** | Pipeline, threshold, feature list and metadata are saved together, so they cannot get out of sync. |

## Project structure

```
credit-card-fraud-detection/
├── src/fraud_detection/
│   ├── config.py        # Paths, feature lists, constants
│   ├── data.py          # Loading, validation, cleaning, splitting
│   ├── modeling.py      # Pipelines, CV comparison, tuning, threshold selection
│   ├── evaluate.py      # Metrics, bootstrap CIs, plots, permutation importance
│   ├── artifacts.py     # Save/load the deployable model file
│   ├── train.py         # End-to-end training CLI
│   └── api.py           # FastAPI service
├── notebooks/
│   └── creditcard_fraud_detection.ipynb   # EDA, reasoning and detailed results
├── tests/               # Unit and API tests (run on synthetic data)
├── data/                # Put creditcard.csv here (git-ignored)
├── models/              # Trained model (git-ignored, created by `make train`)
├── reports/             # metrics.json and figures (created by training)
├── .github/workflows/ci.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Quick start

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install                       # or: pip install -e ".[notebook,dev]"
```

Download `creditcard.csv` from Kaggle into `data/`, then:

```bash
make train          # full run, several minutes: compares models, tunes, evaluates, saves model
make serve          # starts the API at http://127.0.0.1:8000
```

Want to check that everything works without the dataset? `make train-quick` runs the whole pipeline on synthetic data. Its metrics are meaningless and the saved model must not be used for anything.

To explore the analysis, open `notebooks/creditcard_fraud_detection.ipynb`. It imports the same package code.

## API

Interactive documentation is generated at **`/docs`**.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | `200` when the model is loaded, `503` otherwise |
| `GET` | `/model-info` | Model name, version, threshold, features, test metrics |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/predict/batch` | Score up to 1,000 transactions (`{"transactions": [...]}`) |

A transaction needs the 29 numeric fields `V1`–`V28` and `Amount` (≥ 0). **Do not send `Time`.** Missing, extra, non-numeric or non-finite fields are rejected with `422`.

```python
import httpx
import pandas as pd
from fraud_detection.config import FEATURES

row = pd.read_csv("data/creditcard.csv").iloc[0][FEATURES].to_dict()
print(httpx.post("http://127.0.0.1:8000/predict", json=row).json())
```

Response shape:

```json
{
  "is_fraud": false,
  "fraud_score": 0.0031,
  "threshold": 0.18,
  "model_version": "2026-10-23-4590423553c1"
}
```

`is_fraud` is `fraud_score >= threshold`. The model location can be changed with the `MODEL_PATH` environment variable.

## Docker

The image contains code only. The trained model is mounted at runtime.

```bash
make train
docker build -t fraud-detection-api .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" fraud-detection-api
```

The container runs as a non-root user and has a `/health` health check.

## Development

```bash
make test      # pytest (uses synthetic data, no dataset needed)
make lint      # ruff check + format check
make format    # auto-format
```

CI (GitHub Actions) runs lint and tests on Python 3.10, 3.11 and 3.12 and builds the Docker image on every push and pull request.

## Limitations

- **Very few fraud cases.** ~100 positives means every estimate is noisy. Always read metrics together with their confidence intervals.
- **Anonymised features.** `V1`–`V28` are PCA outputs, so domain features (merchant, velocity, geography) cannot be engineered.
- **Random split.** The notebook includes a "train on the past, test on the future" check, but the main evaluation uses a stratified random split.
- **Uncalibrated scores.** `fraud_score` ranks risk but is not a probability.
- **No production monitoring.** A real deployment needs score-drift and alert-rate monitoring, scheduled retraining, and a feedback loop from fraud analysts.
- **Pickle-based model file.** Only load model files you produced yourself.

## License

MIT. See [LICENSE](LICENSE) and replace `<Your Name>` with your name.

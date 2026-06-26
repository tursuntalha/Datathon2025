# Datathon 2025 — Session Value Prediction

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

A machine learning solution built for the **BTK Akademi Datathon 2025** competition hosted on Kaggle. The goal is to predict `session_value` — a continuous target representing the monetary or engagement value of a user session — using user behavior data and feature engineering.

**Project Vision:** This datathon submission explores behavioral session data for predictive modeling. The feature engineering techniques developed here — time-based aggregations, user-level behavioral fingerprints, and session-window statistics — are directly transferable to production systems: real-time personalization engines, dynamic pricing models, and product recommendation APIs all rely on the same core pattern of extracting signal from raw user interaction logs.

---

## Competition Overview

| Field | Detail |
|---|---|
| Organizer | BTK Akademi |
| Platform | Kaggle |
| Task | Regression — predict `session_value` |
| Metric | RMSE (Root Mean Squared Error) |

---

## Problem Statement

Given anonymized user behavior logs for each session, the model must predict a continuous `session_value` score. Sessions include features from user interactions, time patterns, and categorical identifiers. The challenge lies in crafting informative features from raw behavioral signals.

---

## Approach

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │  ─ Distribution analysis
   │  ─ Correlation heatmaps
   │  ─ Missing value inspection
   ▼
Feature Engineering
   │  ─ Time-based features (hour, day of week, weekend flag)
   │  ─ Interaction features
   │  ─ Categorical encoding
   │  ─ Aggregation features per user/session
   ▼
LightGBM Training
   │  ─ K-Fold Cross-Validation
   │  ─ Hyperparameter tuning
   │  ─ Early stopping
   ▼
Submission (lgbm_model_submission.csv)
```

---

## Project Structure

```
Datathon2025/
├── datathon2025.ipynb              # Main notebook: EDA, feature engineering, training
├── features.py                     # Shared feature engineering module
├── utils.py                        # Shared utilities (model I/O, paths)
├── api/
│   └── main.py                     # (1) Real-Time Scoring API (FastAPI)
├── benchmarks/
│   └── neural_benchmark.py         # (2) Neural Approaches Benchmark
├── explainability/
│   └── explain.py                  # (3) SHAP + LIME explainability layer
├── ablation/
│   └── ablation.py                 # (4) Feature Ablation Study
├── drift/
│   └── drift.py                    # (5) Data Drift Detection (Evidently AI)
├── streaming/
│   └── pipeline.py                 # (6) Streaming Feature Pipeline (Redis+Python)
├── datathon_dataset/
│   ├── train.csv                   # Training data with labels
│   ├── test.csv                    # Test data for submission
│   └── sample_submission.csv       # Submission format template
├── outputs/
│   └── lgbm_model_submission.csv   # Generated predictions
└── requirements.txt
```

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook datathon2025.ipynb
```

Run all cells in order. The final cell writes predictions to `outputs/lgbm_model_submission.csv`.

---

## Data Source

This project uses data from the **BTK Akademi Datathon 2025** competition on Kaggle. Data is subject to competition terms and conditions — refer to the official competition page for licensing and usage policies.

---

## Results

The model output is a `submission.csv` file ready for upload to the Kaggle leaderboard. Cross-validation scores and feature importance plots are generated inside the notebook.

---

## Beyond the Datathon

The session value prediction problem is foundational to dozens of real-world applications. Next steps to evolve this into production-grade work:

- [x] **Real-Time Scoring API** — Wrap the trained LightGBM model in a FastAPI endpoint. Input: raw session event stream. Output: predicted session value in <100ms. This could directly power a recommendation engine or dynamic pricing system.
  - `api/main.py`: FastAPI app with `/predict` endpoint, auto-trains model if none exists, returns predictions in ms.
  - Usage: `uvicorn api.main:app --reload`
- [x] **Neural Approaches Benchmark** — Compare LightGBM against TabNet, NODE (Neural Oblivious Decision Ensembles), and FT-Transformer on the same CV setup. Does deep learning help on this tabular problem? Measure and document.
  - `benchmarks/neural_benchmark.py`: Time-based CV loop, TableNet via `pytorch-tabnet`, NODE & FT-Transformer via PyTorch implementations.
  - Usage: `python -m benchmarks.neural_benchmark`
- [x] **Explainability Layer** — Add SHAP waterfall plots for individual session predictions ("This session's value is high because: weekend=True contributed +12.3, user_history_30d contributed +8.7..."). Add LIME as a second explainer for cross-validation.
  - `explainability/explain.py`: SHAP TreeExplainer waterfall + LIME TabularExplainer on sample sessions.
  - Usage: `python -m explainability.explain`
- [x] **Feature Ablation Study** — Systematically remove feature groups (time features, interaction features, aggregation features) and measure CV RMSE degradation. Produces a clean story about what actually drives session value.
  - `ablation/ablation.py`: 14 feature groups defined, each ablated against time-based CV, results ranked by RMSE degradation.
  - Usage: `python -m ablation.ablation`
- [x] **Data Drift Detection** — Integrate Evidently AI to monitor feature distributions between train and test. Build a drift report that would flag if the model is being applied to out-of-distribution sessions in a production deployment.
  - `drift/drift.py`: Evidently AI DataDriftPreset + PSI/K-S fallback if Evidently not installed.
  - Usage: `python -m drift.drift`
- [x] **Streaming Feature Pipeline** — Redesign the feature engineering as a streaming pipeline using Kafka + Flink (or simpler: Redis + Python). Simulate real-time feature computation as events arrive during a live session.
  - `streaming/pipeline.py`: Redis+Python stateful processor, simulates event stream at configurable speed, computes incremental features.
  - Usage: `python -m streaming.pipeline`

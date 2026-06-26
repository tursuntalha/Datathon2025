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
├── datathon2025.ipynb        # Main notebook: EDA, feature engineering, training
├── datathon_dataset/
│   ├── train.csv             # Training data with labels
│   ├── test.csv              # Test data for submission
│   └── sample_submission.csv # Submission format template
├── outputs/
│   └── lgbm_model_submission.csv  # Generated predictions
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

import os
import joblib
import pandas as pd
import lightgbm as lgb
from pathlib import Path

DATA_DIR = "datathon_dataset"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
MODEL_DIR = "outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")


def load_data(train_only=False):
    train = pd.read_csv(TRAIN_PATH)
    if train_only:
        return train
    test = pd.read_csv(TEST_PATH)
    return train, test


def save_model(model, path=None):
    path = path or MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path=None):
    path = path or MODEL_PATH
    return joblib.load(path)


def train_final_model(X, y, cat_cols):
    model = lgb.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        learning_rate=0.1,
        n_estimators=100,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y, categorical_feature=cat_cols)
    return model

import os
import sys
import time
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import engineer_features, get_feature_columns, cat_cols
from utils import load_model, train_final_model, load_data, save_model

app = FastAPI(title="Session Value Prediction API", version="1.0.0")

model = None
feature_cols = None


class Event(BaseModel):
    event_time: str
    event_type: str
    product_id: str
    category_id: str
    user_id: str
    user_session: str


class PredictRequest(BaseModel):
    events: List[Event]


class PredictResponse(BaseModel):
    user_session: str
    predicted_value: float
    inference_ms: float


def ensure_model():
    global model, feature_cols
    if model is not None:
        return
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "lgbm_model.pkl")
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("No saved model found. Training on full dataset...")
        train_df, _ = load_data(train_only=False)
        train_df = engineer_features(train_df)
        feature_cols_all = get_feature_columns(train_df)
        feature_cols = [c for c in feature_cols_all if c in train_df.columns]
        X = train_df[feature_cols]
        y = train_df["session_value"]
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        model = train_final_model(X, y, cat_cols)
        save_model(model, model_path)
    if feature_cols is None:
        train_df, _ = load_data(train_only=False)
        train_df = engineer_features(train_df)
        feature_cols_all = get_feature_columns(train_df)
        feature_cols = [c for c in feature_cols_all if c in train_df.columns]


@app.on_event("startup")
async def startup():
    ensure_model()


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    ensure_model()
    t0 = time.time()
    rows = [e.dict() for e in request.events]
    df = pd.DataFrame(rows)
    try:
        df_feat = engineer_features(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {e}")
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        for c in missing:
            df_feat[c] = 0
    X = df_feat[feature_cols]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    preds = model.predict(X)
    session_pred = float(np.mean(preds))
    elapsed_ms = (time.time() - t0) * 1000
    session_id = rows[0]["user_session"] if rows else "unknown"
    return PredictResponse(
        user_session=session_id,
        predicted_value=round(session_pred, 4),
        inference_ms=round(elapsed_ms, 2),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

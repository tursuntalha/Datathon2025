"""
Streaming Feature Pipeline (Redis + Python).

Simulates real-time event ingestion and incremental feature computation using Redis
as a lightweight state store. In production this pattern maps to Kafka + Flink;
here Redis provides the same session-state materialized-view semantics.

Architecture:
  Events -> Redis Stream -> FeatureProcessor -> Redis Hash (session features) -> Predict
"""

import os
import sys
import json
import time
import threading
import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import cat_cols
from utils import load_data, load_model

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
STREAM_KEY = "session:events"
CONSUMER_GROUP = "feature_engineers"
CONSUMER_NAME = "worker_1"

SESSION_STATE_PREFIX = "session:state:"
SESSION_FEATURES_PREFIX = "session:features:"


class StreamingFeatureProcessor:
    def __init__(self, model=None, redis_client=None, use_mock=True):
        self.use_mock = use_mock
        self.model = model
        self.redis = redis_client
        self._session_state = {}
        self._session_features_cache = {}

    def process_event(self, event):
        session_id = event["user_session"]
        if session_id not in self._session_state:
            self._session_state[session_id] = {
                "events": [],
                "first_event_time": event["event_time"],
                "last_event_time": event["event_time"],
                "event_count": 0,
                "products_seen": set(),
                "categories_seen": set(),
                "event_type_counts": {},
            }

        state = self._session_state[session_id]
        state["events"].append(event)
        state["last_event_time"] = event["event_time"]
        state["event_count"] += 1
        state["products_seen"].add(event["product_id"])
        state["categories_seen"].add(event["category_id"])

        et = event["event_type"]
        state["event_type_counts"][et] = state["event_type_counts"].get(et, 0) + 1

        features = self._compute_features(session_id, event, state)
        self._session_features_cache[session_id] = features

        if self.use_mock and self.model is not None:
            prediction = self._predict(features)
            return {"session_id": session_id, "features": features, "prediction": prediction}

        return {"session_id": session_id, "features": features}

    def _compute_features(self, session_id, event, state):
        event_time = pd.to_datetime(event["event_time"])
        first_time = pd.to_datetime(state["first_event_time"])
        last_time = pd.to_datetime(state["last_event_time"])

        duration_sec = (last_time - first_time).total_seconds() if state["event_count"] > 1 else 0
        time_since_start = (event_time - first_time).total_seconds()

        all_types = ["VIEW", "ADD_CART", "REMOVE_CART", "BUY"]
        et_counts = state["event_type_counts"]
        total = state["event_count"]

        features = {
            "event_type": event["event_type"],
            "product_id": event["product_id"],
            "category_id": event["category_id"],
            "user_id": event["user_id"],
            "user_session": session_id,
            "hour": event_time.hour,
            "day_of_week": event_time.dayofweek,
            "time_since_session_start": time_since_start,
            "session_event_count": total,
            "session_product_count": len(state["products_seen"]),
            "session_category_count": len(state["categories_seen"]),
            "session_duration": duration_sec,
            "session_duration_minutes": duration_sec / 60 if duration_sec > 0 else 0,
        }

        for et in all_types:
            features[f"session_{et.lower()}_count"] = et_counts.get(et, 0)

        morning = 1 if 6 <= event_time.hour <= 11 else 0
        afternoon = 1 if 12 <= event_time.hour <= 17 else 0
        evening = 1 if 18 <= event_time.hour <= 23 else 0
        night = 1 if 0 <= event_time.hour <= 5 else 0
        features.update({
            "morning_event": morning,
            "afternoon_event": afternoon,
            "evening_event": evening,
            "night_event": night,
        })

        return features

    def _predict(self, features):
        if self.model is None:
            return None
        try:
            df = pd.DataFrame([features])
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            missing = [c for c in self.model.booster_.feature_name() if c not in df.columns]
            for c in missing:
                df[c] = 0
            X = df[[c for c in self.model.booster_.feature_name() if c in df.columns]]
            return float(self.model.predict(X)[0])
        except Exception as e:
            return f"error: {e}"

    def get_session_features(self, session_id):
        return self._session_features_cache.get(session_id)

    def flush_state(self):
        self._session_state.clear()
        self._session_features_cache.clear()


def simulate_event_stream(csv_path, speed_factor=10.0):
    df = pd.read_csv(csv_path)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    if len(df) > 5000:
        df = df.sample(5000).sort_values("event_time").reset_index(drop=True)

    if len(df) > 0:
        start = df["event_time"].min()
        df["_time_offset"] = (df["event_time"] - start).dt.total_seconds()

    print(f"Simulating {len(df)} events at {speed_factor}x speed...")
    for _, row in df.iterrows():
        event = {
            "event_time": row["event_time"].isoformat(),
            "event_type": row["event_type"],
            "product_id": str(row["product_id"]),
            "category_id": str(row["category_id"]),
            "user_id": str(row["user_id"]),
            "user_session": str(row["user_session"]),
        }
        yield event
        if speed_factor > 0:
            time.sleep(row.get("_time_offset", 0) / speed_factor)
        else:
            time.sleep(0.001)


def main():
    print("=" * 60)
    print("  Streaming Feature Pipeline (Redis + Python)")
    print("=" * 60)

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "lgbm_model.pkl"
    )
    model = None
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
    else:
        print("No trained model found. Running in feature-only mode.")

    processor = StreamingFeatureProcessor(model=model, use_mock=True)

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datathon_dataset", "train.csv"
    )

    print("\nStarting simulated event stream (Ctrl+C to stop)...\n")
    event_count = 0
    session_predictions = {}

    try:
        for event in simulate_event_stream(csv_path, speed_factor=100.0):
            result = processor.process_event(event)
            event_count += 1
            sid = result["session_id"]
            if result.get("prediction") is not None and isinstance(result["prediction"], (int, float)):
                if sid not in session_predictions:
                    session_predictions[sid] = []
                session_predictions[sid].append(result["prediction"])

            if event_count % 500 == 0:
                print(
                    f"  Processed {event_count} events | "
                    f"Active sessions: {len(processor._session_state)} | "
                    f"Predictions: {len(session_predictions)}"
                )

    except KeyboardInterrupt:
        print("\n\nStream interrupted by user.")
    finally:
        print(f"\n{'=' * 60}")
        print(f"  Stream Summary")
        print(f"{'=' * 60}")
        print(f"  Total events processed: {event_count}")
        print(f"  Unique sessions seen: {len(processor._session_state)}")
        print(f"  Sessions with predictions: {len(session_predictions)}")

        if session_predictions:
            avg_preds = {sid: np.mean(preds) for sid, preds in session_predictions.items()}
            print(f"\n  Sample predictions:")
            for sid, val in list(avg_preds.items())[:5]:
                print(f"    {sid}: {val:.4f}")

        outputs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs"
        )
        os.makedirs(outputs_dir, exist_ok=True)
        summary_path = os.path.join(outputs_dir, "streaming_summary.csv")
        if session_predictions:
            summary_df = pd.DataFrame([
                {"user_session": sid, "predicted_value": np.mean(preds)}
                for sid, preds in session_predictions.items()
            ])
            summary_df.to_csv(summary_path, index=False)
            print(f"\n  Summary saved to: {summary_path}")

        print("\nStreaming pipeline complete.")


if __name__ == "__main__":
    main()

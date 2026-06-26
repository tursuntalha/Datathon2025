import os
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import engineer_features, get_feature_columns, cat_cols
from utils import load_data

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

FEATURE_GROUPS = {
    "time_features": [
        "day_of_week", "hour", "morning_event", "afternoon_event",
        "evening_event", "night_event",
    ],
    "session_features": [
        "time_since_session_start", "session_event_count", "session_product_count",
        "session_category_count", "session_duration", "session_duration_minutes",
        "session_unique_days", "session_day_span", "session_daily_avg_events",
        "time_to_first_add_cart", "time_to_first_buy",
        "mean_inter_event_sec", "std_inter_event_sec",
        "session_duration_sec", "mean_inter_event_sec_y", "std_inter_event_sec_y",
        "unique_days",
    ],
    "session_event_type_features": [
        "ADD_CART", "VIEW", "REMOVE_CART", "BUY",
        "add_cart_rate", "view_rate", "remove_cart_rate", "buy_rate",
        "view_to_add_cart_ratio", "view_to_remove_cart_ratio",
        "add_cart_to_buy_ratio", "add_cart_to_remove_cart_ratio",
        "total_events", "mean_events", "std_events", "max_events", "min_events",
        "morning_events", "afternoon_events", "evening_events", "night_events",
        "unique_products", "unique_categories",
    ],
    "user_features": [
        "user_total_events", "user_unique_sessions", "user_unique_products",
        "user_unique_categories", "user_total_duration", "user_unique_days",
        "user_avg_events_per_day", "user_most_active_day", "user_days_range",
        "user_avg_time_between_events", "user_std_time_between_events",
        "user_avg_session_duration",
    ],
    "user_event_type_features": [
        "user_view_count", "user_add_cart_count", "user_remove_cart_count",
        "user_buy_count", "user_view_rate", "user_add_cart_rate",
        "user_remove_cart_rate", "user_buy_rate",
        "user_view_to_add_cart_ratio", "user_add_cart_to_buy_ratio",
        "user_buy_to_total_ratio", "user_cart_abandon_ratio",
    ],
    "product_features": [
        "product_total_events", "product_unique_users", "product_unique_sessions",
        "product_unique_categories", "product_total_duration",
        "product_avg_time_between_events", "product_std_time_between_events",
    ],
    "product_event_type_features": [
        "product_view_count", "product_add_cart_count", "product_remove_cart_count",
        "product_buy_count", "product_view_rate", "product_add_cart_rate",
        "product_remove_cart_rate", "product_buy_rate",
        "product_view_to_add_cart_ratio", "product_add_cart_to_buy_ratio",
        "product_buy_to_total_ratio", "product_cart_abandon_ratio",
        "product_buy_count_y", "product_view_count_y",
    ],
    "category_features": [
        "category_total_events", "category_unique_users", "category_unique_sessions",
        "category_unique_products", "category_total_duration",
        "category_avg_time_between_events", "category_std_time_between_events",
    ],
    "category_event_type_features": [
        "category_view_count", "category_add_cart_count", "category_remove_cart_count",
        "category_buy_count", "category_view_rate", "category_add_cart_rate",
        "category_remove_cart_rate", "category_buy_rate",
        "category_view_to_add_cart_ratio", "category_add_cart_to_buy_ratio",
        "category_buy_to_total_ratio", "category_cart_abandon_ratio",
        "category_add_cart_count_y", "category_view_count_y",
        "category_buy_count_y", "category_remove_cart_count_y",
    ],
    "daily_features": [
        "daily_event_count", "daily_unique_users", "daily_unique_products",
        "daily_add_cart_count", "daily_view_count", "daily_remove_cart_count",
        "daily_buy_count", "daily_add_cart_rate", "daily_view_rate",
        "daily_remove_cart_rate", "daily_buy_rate",
    ],
    "session_to_daily_features": [
        "session_to_daily_add_cart_ratio", "session_to_daily_view_ratio",
        "session_to_daily_remove_cart_ratio", "session_to_daily_buy_ratio",
    ],
    "count_features": [
        "user_view_count_x", "user_add_cart_count_x", "user_remove_cart_count_x",
        "user_buy_count_x", "category_view_count_x", "category_add_cart_count_x",
        "category_remove_cart_count_x", "category_buy_count_x",
        "product_view_count_x", "product_add_cart_count_x",
        "product_remove_cart_count_x", "product_buy_count_x",
    ],
}


def time_based_cv(train_df, feature_cols, n_folds=4, train_days=9, val_days=3):
    train_df = train_df.sort_values("event_time").reset_index(drop=True)
    min_date = train_df["event_time"].min().normalize()
    max_date = train_df["event_time"].max().normalize()
    total_days = (max_date - min_date).days + 1
    scores = []
    start_day = 0
    fold_idx = 0
    while start_day + train_days + val_days <= total_days:
        train_start = min_date + pd.Timedelta(days=start_day)
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=val_days - 1)

        train_mask = (train_df["event_time"].dt.date >= train_start.date()) & (
            train_df["event_time"].dt.date <= train_end.date()
        )
        val_mask = (train_df["event_time"].dt.date >= val_start.date()) & (
            train_df["event_time"].dt.date <= val_end.date()
        )

        X_tr = train_df[train_mask][feature_cols]
        y_tr = train_df[train_mask]["session_value"]
        X_va = train_df[val_mask][feature_cols]
        y_va = train_df[val_mask]["session_value"]

        if X_va.empty:
            start_day += val_days
            continue

        model = lgb.LGBMRegressor(
            objective="regression", boosting_type="gbdt",
            learning_rate=0.1, n_estimators=100,
            num_leaves=31, max_depth=5,
            random_state=42, verbose=-1,
        )
        available_cat = [c for c in cat_cols if c in feature_cols]
        model.fit(X_tr, y_tr, categorical_feature=available_cat)
        preds = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        scores.append(rmse)
        start_day += val_days
        fold_idx += 1

    return np.mean(scores), np.std(scores)


def main():
    print("=" * 60)
    print("  Feature Ablation Study")
    print("=" * 60)

    train_df, _ = load_data()
    print(f"\nLoaded {len(train_df)} training rows. Engineering features...")
    train_df = engineer_features(train_df)
    all_feature_cols = get_feature_columns(train_df)
    all_feature_cols = [c for c in all_feature_cols if c in train_df.columns]

    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("category")

    print(f"Total features: {len(all_feature_cols)}")
    print("\nComputing baseline RMSE (all features)...")
    baseline_mean, baseline_std = time_based_cv(train_df, all_feature_cols)
    print(f"  Baseline RMSE: {baseline_mean:.4f} +/- {baseline_std:.4f}\n")

    results = []
    results.append({
        "feature_group": "ALL FEATURES (baseline)",
        "n_features": len(all_feature_cols),
        "mean_rmse": round(baseline_mean, 4),
        "std_rmse": round(baseline_std, 4),
        "rmse_degradation": 0.0,
        "degradation_pct": 0.0,
    })

    for group_name, group_features in FEATURE_GROUPS.items():
        present = [f for f in group_features if f in all_feature_cols]
        if not present:
            print(f"  Skipping '{group_name}': no matching features found")
            continue

        remaining = [f for f in all_feature_cols if f not in present]
        print(f"  Ablating '{group_name}' ({len(present)} features)...")
        mean_rmse, std_rmse = time_based_cv(train_df, remaining)

        degradation = mean_rmse - baseline_mean
        degradation_pct = (degradation / baseline_mean) * 100 if baseline_mean > 0 else 0

        results.append({
            "feature_group": group_name,
            "n_features": len(remaining),
            "mean_rmse": round(mean_rmse, 4),
            "std_rmse": round(std_rmse, 4),
            "rmse_degradation": round(degradation, 4),
            "degradation_pct": round(degradation_pct, 2),
        })
        sign = "+" if degradation > 0 else ""
        print(
            f"    RMSE: {mean_rmse:.4f} ({sign}{degradation:.4f}, {sign}{degradation_pct:.2f}%)\n"
        )

    res_df = pd.DataFrame(results).sort_values("rmse_degradation", ascending=False)
    print("\n" + "=" * 60)
    print("  Ablation Study Summary (sorted by degradation)")
    print("=" * 60)
    print(
        f"  {'Feature Group':35s} {'#Feat':6s} {'RMSE':10s} {'Change':10s} {'%Change':8s}"
    )
    print(f"  {'-' * 71}")
    for _, row in res_df.iterrows():
        ch = row["rmse_degradation"]
        sign = "+" if ch > 0 else ""
        print(
            f"  {row['feature_group']:35s} {int(row['n_features']):6d} {row['mean_rmse']:10.4f} {sign}{ch:<9.4f} {sign}{row['degradation_pct']:<7.2f}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("Ablation study complete.")


if __name__ == "__main__":
    main()

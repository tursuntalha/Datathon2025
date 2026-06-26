import os
import sys
import warnings
import json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import engineer_features, get_feature_columns
from utils import load_data

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def detect_drift_evidently(
    reference_df, current_df, numerical_features, categorical_features, output_path
):
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.test_preset import DataStabilityTestPreset
    except ImportError:
        return _detect_drift_fallback(reference_df, current_df, numerical_features, output_path)

    ref = reference_df[numerical_features + categorical_features].copy()
    cur = current_df[numerical_features + categorical_features].copy()

    for c in categorical_features:
        if c in ref.columns:
            ref[c] = ref[c].astype(str)
        if c in cur.columns:
            cur[c] = cur[c].astype(str)

    report = Report(metrics=[DataDriftPreset(), DataStabilityTestPreset()])
    report.run(reference_data=ref, current_data=cur)

    report_path = output_path.replace(".json", "_evidently_report.html")
    report.save_html(report_path)
    print(f"Evidently report saved to: {report_path}")

    summary = report.as_dict()
    drift_metrics = {}
    for metric in summary.get("metrics", []):
        if "result" in metric:
            drift_metrics.update(metric["result"])

    json_path = output_path.replace(".json", "_evidently_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Evidently metrics saved to: {json_path}")

    return summary


def _detect_drift_fallback(reference_df, current_df, numerical_features, output_path):
    print("Evidently AI not installed. Using fallback PSI-based drift detection.")
    print("Install with: pip install evidently")

    def calculate_psi(expected, actual, bins=10):
        expected = np.clip(expected, 1e-10, None)
        actual = np.clip(actual, 1e-10, None)
        psi = np.sum((actual - expected) * np.log(actual / expected))
        return psi

    drift_results = []
    for feat in numerical_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue

        ref_vals = reference_df[feat].dropna()
        cur_vals = current_df[feat].dropna()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        try:
            bin_edges = np.percentile(ref_vals, np.linspace(0, 100, bins + 1))
            bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)

            ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
            cur_counts, _ = np.histogram(cur_vals, bins=bin_edges)

            ref_pct = ref_counts / len(ref_vals)
            cur_pct = cur_counts / len(cur_vals)

            psi = calculate_psi(ref_pct, cur_pct)

            ks_stat = None
            try:
                from scipy.stats import ks_2samp
                ks_stat, ks_pval = ks_2samp(ref_vals, cur_vals)
            except ImportError:
                ks_stat = None

            drift_detected = psi > 0.2 or (ks_stat is not None and ks_stat > 0.1)
            drift_results.append({
                "feature": feat,
                "psi": round(psi, 6),
                "ks_statistic": round(ks_stat, 6) if ks_stat is not None else None,
                "drift_detected": drift_detected,
                "ref_mean": round(ref_vals.mean(), 4),
                "cur_mean": round(cur_vals.mean(), 4),
                "ref_std": round(ref_vals.std(), 4),
                "cur_std": round(cur_vals.std(), 4),
            })
        except Exception as e:
            print(f"  Error on {feat}: {e}")

    return drift_results


def main():
    print("=" * 60)
    print("  Data Drift Detection (Evidently AI)")
    print("=" * 60)

    train_df, test_df = load_data()
    print(f"\nLoaded {len(train_df)} train rows, {len(test_df)} test rows.")
    print("Engineering features for both sets...")

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    feature_cols_all = get_feature_columns(train_df)
    feature_cols = [c for c in feature_cols_all if c in train_df.columns and c in test_df.columns]

    numerical_features = train_df[feature_cols].select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numerical_features]

    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    output_path = os.path.join(OUTPUT_DIR, "drift_report.json")
    result = detect_drift_evidently(
        train_df, test_df, numerical_features, categorical_features, output_path
    )

    if isinstance(result, list):
        drift_count = sum(1 for r in result if r.get("drift_detected"))
        print(f"\n{'=' * 60}")
        print(f"  Drift Detection Summary (PSI/K-S Fallback)")
        print(f"{'=' * 60}")
        print(f"  Features with drift: {drift_count}/{len(result)}")

        print(f"\n  {'Feature':30s} {'PSI':10s} {'Drift':8s}")
        print(f"  {'-' * 50}")
        for r in sorted(result, key=lambda x: x["psi"], reverse=True)[:20]:
            flag = "DRIFT" if r["drift_detected"] else "OK"
            print(f"  {r['feature']:30s} {r['psi']:<10.6f} {flag:8s}")

        res_path = os.path.join(OUTPUT_DIR, "drift_results.csv")
        pd.DataFrame(result).to_csv(res_path, index=False)
        print(f"\nResults saved to: {res_path}")
    else:
        print("\nFull Evidently report generated with HTML visualization.")

    print("\nDrift detection complete.")


if __name__ == "__main__":
    main()

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import engineer_features, get_feature_columns, cat_cols
from utils import load_model, load_data

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def shap_explain(model, X_sample, feature_names, output_path=None):
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0],
        "feature_value": X_sample.iloc[0].values,
    }).sort_values("shap_value", key=abs, ascending=False)

    print("\nSHAP Waterfall (top 10 contributors):")
    print(f"  Base value (expected): {explainer.expected_value:.4f}")
    print(f"  Model prediction: {model.predict(X_sample)[0]:.4f}")
    print(f"  {'Feature':30s} {'Value':12s} {'SHAP Contribution':20s}")
    print(f"  {'-' * 62}")
    for _, row in shap_df.head(10).iterrows():
        sign = "+" if row["shap_value"] > 0 else ""
        print(
            f"  {str(row['feature']):30s} {str(row['feature_value']):12s} {sign}{row['shap_value']:.4f}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sample.iloc[0].values,
                feature_names=feature_names,
            ),
            show=False,
            max_display=15,
        )
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"\nWaterfall plot saved to: {output_path}")
    except Exception as e:
        print(f"  (Plotting skipped: {e})")

    return shap_df


def lime_explain(model, X_sample, feature_names, output_path=None):
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print("LIME not installed. Install with: pip install lime")
        return None

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_sample.values,
        feature_names=feature_names,
        mode="regression",
        random_state=42,
    )
    exp = explainer.explain_instance(
        X_sample.iloc[0].values,
        model.predict,
        num_features=10,
    )

    print("\nLIME Explanation (top 10 features):")
    print(f"  {'Feature':30s} {'Contribution':20s}")
    print(f"  {'-' * 50}")
    for feat, contrib in exp.as_list():
        sign = "+" if contrib > 0 else ""
        print(f"  {str(feat):30s} {sign}{contrib:.4f}")

    if output_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = exp.as_pyplot_figure()
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"LIME plot saved to: {output_path}")
        except Exception as e:
            print(f"  (Plotting skipped: {e})")

    return exp


def explain_session(session_id, train_df, model, feature_cols, output_dir=OUTPUT_DIR):
    session_data = train_df[train_df["user_session"] == session_id]
    if session_data.empty:
        print(f"Session {session_id} not found.")
        return

    X_session = session_data[feature_cols].iloc[:1]
    pred = model.predict(X_session)[0]
    print(f"\nSession: {session_id}")
    print(f"Predicted session_value: {pred:.4f}")

    shap_path = os.path.join(output_dir, f"shap_waterfall_{session_id}.png")
    lime_path = os.path.join(output_dir, f"lime_explanation_{session_id}.png")

    print("\n" + "=" * 50)
    print("SHAP Analysis")
    print("=" * 50)
    shap_explain(model, X_session, feature_cols, output_path=shap_path)

    print("\n" + "=" * 50)
    print("LIME Analysis")
    print("=" * 50)
    lime_explain(model, X_session, feature_cols, output_path=lime_path)

    print("\nDone. Both SHAP and LIME explanations generated.")


def main():
    print("=" * 50)
    print("  Explainability Layer (SHAP + LIME)")
    print("=" * 50)

    train_df, _ = load_data()
    train_df = engineer_features(train_df)
    feature_cols_all = get_feature_columns(train_df)
    feature_cols = [c for c in feature_cols_all if c in train_df.columns]

    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("category")

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "lgbm_model.pkl")
    if not os.path.exists(model_path):
        print("No trained model found. Training final model...")
        from utils import train_final_model
        X = train_df[feature_cols]
        y = train_df["session_value"]
        model = train_final_model(X, y, cat_cols)
        from utils import save_model
        save_model(model, model_path)
    else:
        model = load_model(model_path)

    sample_sessions = train_df["user_session"].unique()[:3]
    for sid in sample_sessions:
        explain_session(sid, train_df, model, feature_cols)
        print("\n" + "-" * 70)

    all_shap = []
    for sid in sample_sessions:
        session_data = train_df[train_df["user_session"] == sid]
        X_s = session_data[feature_cols].iloc[:1]
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_s)
            for j, fn in enumerate(feature_cols):
                all_shap.append({"session": sid, "feature": fn, "shap_value": sv[0][j]})
        except ImportError:
            break

    if all_shap:
        summary_path = os.path.join(OUTPUT_DIR, "shap_summary.csv")
        pd.DataFrame(all_shap).to_csv(summary_path, index=False)
        print(f"\nSHAP summary saved to: {summary_path}")

    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import engineer_features, get_feature_columns, cat_cols
from utils import load_data

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def time_based_cv_split(train_df, n_folds=4, train_days=9, val_days=3):
    train_df = train_df.sort_values("event_time").reset_index(drop=True)
    min_date = train_df["event_time"].min().normalize()
    max_date = train_df["event_time"].max().normalize()
    total_days = (max_date - min_date).days + 1
    folds = []
    start_day = 0
    while start_day + train_days + val_days <= total_days:
        train_start = min_date + pd.Timedelta(days=start_day)
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=val_days - 1)
        folds.append((train_start, train_end, val_start, val_end))
        start_day += val_days
    return folds, train_df["event_time"]


def run_lightgbm(X_train, y_train, X_val, y_val, cat_features):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        learning_rate=0.1,
        n_estimators=100,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=cat_features)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds)), model


def run_tabnet(X_train, y_train, X_val, y_val, cat_features):
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        return None, None

    unsupervised_model = None
    model = TabNetRegressor(
        n_d=16, n_a=16, n_steps=3, gamma=1.5,
        lambda_sparse=1e-4, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2), mask_type="sparsemax",
        scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=0,
    )
    cat_idxs = []
    cat_dims = []
    for i, col in enumerate(X_train.columns):
        if col in cat_features:
            cat_idxs.append(i)
            cat_dims.append(int(X_train[col].nunique()))

    X_train_t = X_train.select_dtypes(include=[np.number]).values.astype(np.float32)
    X_val_t = X_val.select_dtypes(include=[np.number]).values.astype(np.float32)

    model.fit(
        X_train_t, y_train.values.reshape(-1, 1),
        eval_set=[(X_val_t, y_val.values.reshape(-1, 1))],
        max_epochs=50, patience=10, batch_size=256,
        virtual_batch_size=128,
    )
    preds = model.predict(X_val_t).flatten()
    return np.sqrt(mean_squared_error(y_val, preds)), model


def run_node(X_train, y_train, X_val, y_val, cat_features):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return None, None

    class NODELayer(nn.Module):
        def __init__(self, input_dim, num_trees=4, depth=3, output_dim=1):
            super().__init__()
            self.num_trees = num_trees
            self.depth = depth
            self.num_leaves = 2 ** depth
            self.tree_indices = torch.arange(num_trees).unsqueeze(1)

            self.feature_indices = nn.Parameter(torch.randn(num_trees, depth, input_dim), requires_grad=True)
            self.thresholds = nn.Parameter(torch.randn(num_trees, depth), requires_grad=True)
            self.leaf_values = nn.Parameter(torch.randn(num_trees, self.num_leaves, output_dim), requires_grad=True)

        def forward(self, x):
            batch_size = x.shape[0]
            x_expanded = x.unsqueeze(1).unsqueeze(2)
            feat_selections = torch.softmax(self.feature_indices, dim=-1)
            selected_features = (x_expanded * feat_selections.unsqueeze(0)).sum(dim=-1)
            decisions = torch.sigmoid((selected_features - self.thresholds.unsqueeze(0)) * 1.0)
            leaf_indices = (decisions * (2 ** torch.arange(self.depth, device=x.device).float())).sum(dim=2).long()
            leaf_values = self.leaf_values[self.tree_indices.squeeze(), leaf_indices.clamp(0, self.num_leaves - 1)]
            return leaf_values.mean(dim=1).squeeze(-1)

    X_train_t = torch.tensor(X_train.select_dtypes(include=[np.number]).values.astype(np.float32))
    y_train_t = torch.tensor(y_train.values.astype(np.float32))
    X_val_t = torch.tensor(X_val.select_dtypes(include=[np.number]).values.astype(np.float32))
    y_val_t = torch.tensor(y_val.values.astype(np.float32))

    input_dim = X_train_t.shape[1]
    model = NODELayer(input_dim, num_trees=8, depth=3, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).numpy()
    return np.sqrt(mean_squared_error(y_val, preds)), model


def run_ft_transformer(X_train, y_train, X_val, y_val, cat_features):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return None, None

    class FTTransformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1):
            super().__init__()
            self.projection = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True,
                dim_feedforward=256, dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim),
            )

        def forward(self, x):
            x = self.projection(x).unsqueeze(1)
            x = self.transformer(x)
            return self.head(x.squeeze(1)).squeeze(-1)

    X_train_t = torch.tensor(X_train.select_dtypes(include=[np.number]).values.astype(np.float32))
    y_train_t = torch.tensor(y_train.values.astype(np.float32))
    X_val_t = torch.tensor(X_val.select_dtypes(include=[np.number]).values.astype(np.float32))
    y_val_t = torch.tensor(y_val.values.astype(np.float32))

    input_dim = X_train_t.shape[1]
    model = FTTransformer(input_dim, d_model=64, nhead=4, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).numpy()
    return np.sqrt(mean_squared_error(y_val, preds)), model


def main():
    print("=" * 60)
    print("  Neural Approaches Benchmark")
    print("=" * 60)

    train_df, _ = load_data()
    print(f"\nLoaded {len(train_df)} training rows. Engineering features...")
    train_df = engineer_features(train_df)
    feature_cols_all = get_feature_columns(train_df)
    feature_cols = [c for c in feature_cols_all if c in train_df.columns]
    print(f"Feature count: {len(feature_cols)}")

    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("category")

    folds, _ = time_based_cv_split(train_df)
    print(f"Created {len(folds)} time-based folds\n")

    benchmarks = {
        "LightGBM": run_lightgbm,
        "TabNet": run_tabnet,
        "NODE": run_node,
        "FT-Transformer": run_ft_transformer,
    }

    results = {name: [] for name in benchmarks}

    for i, (train_start, train_end, val_start, val_end) in enumerate(folds):
        print(f"  Fold {i + 1}/{len(folds)}: train {train_start.date()} -> {val_end.date()}")

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
            print(f"    Skipping fold {i + 1}: empty validation set")
            continue

        for name, runner in benchmarks.items():
            try:
                rmse, _ = runner(X_tr, y_tr, X_va, y_va, cat_cols)
                if rmse is not None:
                    results[name].append(rmse)
                    print(f"    {name:20s}  RMSE: {rmse:.4f}")
                else:
                    print(f"    {name:20s}  SKIPPED (import error)")
            except Exception as e:
                print(f"    {name:20s}  ERROR: {e}")

    print("\n" + "=" * 60)
    print("  Benchmark Results Summary")
    print("=" * 60)
    summary_rows = []
    for name, rmses in results.items():
        if rmses:
            avg = np.mean(rmses)
            std = np.std(rmses)
            summary_rows.append({
                "Model": name,
                "Avg RMSE": round(avg, 4),
                "Std RMSE": round(std, 4),
                "Folds": len(rmses),
            })
            print(f"  {name:20s}  Avg RMSE: {avg:.4f} ± {std:.4f}  ({len(rmses)} folds)")
        else:
            summary_rows.append({"Model": name, "Avg RMSE": None, "Std RMSE": None, "Folds": 0})
            print(f"  {name:20s}  No results")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "neural_benchmark_results.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nResults saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    import torch
    main()

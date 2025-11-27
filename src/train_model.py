"""
Train a first virtual-sensor model for NOx_EO using XGBoost,
and generate metrics + feature importance + SHAP plots.

Usage:
    conda activate virtual-sensor
    cd ~/projects/virtual-sensor-analysis
    python src/train_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor
import shap

# ---------- Paths ----------

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

CLEAN_PATH = DATA_PROCESSED / "engine_clean.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Project root:", ROOT)
    print("Processed data:", CLEAN_PATH)
    print("Reports       :", REPORT_DIR)
    print()

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned data file not found at: {CLEAN_PATH}")

    # ---------- Load cleaned data ----------
    print(f"Loading cleaned dataset from: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    print("Cleaned shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print()

    # ---------- Define target and features ----------
    target_col = "NOx_EO"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in cleaned data.")

    # Drop rows where target is NaN (safety)
    df = df.dropna(subset=[target_col])

    # Features: all columns except target and Log Point
    feature_cols = [
        c for c in df.columns
        if c != target_col and c != "Log Point"
    ]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Using {len(feature_cols)} features to predict '{target_col}'.")
    print()

    # ---------- Train/validation split ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Val shape  :", X_val.shape)
    print()

    # ---------- XGBoost model ----------
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    print("Training complete.")
    print()

    # ---------- Evaluation ----------
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)

    # Your sklearn version does not support squared=False, so we do sqrt manually
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_val = np.sqrt(mse_val)

    metrics_text = (
        f"Target: {target_col}\n"
        f"Features: {len(feature_cols)}\n"
        f"Train R2:  {r2_train:.4f}\n"
        f"Val R2:    {r2_val:.4f}\n"
        f"Train RMSE:{rmse_train:.4f}\n"
        f"Val RMSE:  {rmse_val:.4f}\n"
    )

    print(metrics_text)

    # Save metrics to file
    metrics_out = REPORT_DIR / "model_metrics_NOx_EO.txt"
    with open(metrics_out, "w") as f:
        f.write(metrics_text)
    print(f"Saved metrics to: {metrics_out}")

    # ---------- Feature importance plot ----------
    importances = model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]  # descending
    top_k = min(15, len(feature_cols))
    top_idx = idx_sorted[:top_k]

    top_features = [feature_cols[i] for i in top_idx]
    top_importances = importances[top_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Feature importance")
    plt.title(f"Top {top_k} Features for {target_col} (XGBoost)")
    plt.tight_layout()

    fi_out = REPORT_DIR / "feature_importance_NOx_EO.png"
    plt.savefig(fi_out, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {fi_out}")

    # ---------- SHAP analysis ----------
    print("\nComputing SHAP values (this may take a bit)...")

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        shap_fig = plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title(f"SHAP Summary Plot for {target_col}")
        shap_out = REPORT_DIR / "shap_beeswarm_NOx_EO.png"
        plt.tight_layout()
        plt.savefig(shap_out, dpi=300)
        plt.close(shap_fig)

        print(f"Saved SHAP beeswarm plot to: {shap_out}")
    except Exception as e:
        print("SHAP analysis failed with error:")
        print(e)
        shap_log = REPORT_DIR / "shap_error_log.txt"
        with open(shap_log, "w") as f:
            f.write(str(e))
        print(f"Saved SHAP error log to: {shap_log}")

    print("\nModel training + analysis Completed Successfully ✓")


if __name__ == "__main__":
    main()


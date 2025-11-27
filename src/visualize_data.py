"""
Visualization suite for the virtual sensor project.

Generates:
  - NOx distribution (hist + KDE)
  - Scatter plots: Rail Pressure, EGR_Rate, q_MI vs NOx_EO
  - Predicted vs True NOx (XGBoost)
  - Residuals vs Predicted NOx
  - SHAP bar plot for top features

Usage:
    conda activate virtual-sensor
    cd ~/projects/virtual-sensor-analysis
    python src/visualize_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor
import shap

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
VIS_DIR = REPORT_DIR / "visualizations"

CLEAN_PATH = DATA_PROCESSED / "engine_clean.csv"

VIS_DIR.mkdir(parents=True, exist_ok=True)


def load_clean_data():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned data file not found at: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    return df


# ---------------------------------------------------------------------
# Basic distributions and scatter plots
# ---------------------------------------------------------------------

def plot_nox_distribution(df, target_col="NOx_EO"):
    if target_col not in df.columns:
        print(f"[WARN] Target column {target_col} not found; skipping NOx distribution.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True)
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.title(f"{target_col} Distribution")
    plt.tight_layout()
    out_path = VIS_DIR / f"{target_col}_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved NOx distribution plot to: {out_path}")


def scatter_vs_nox(df, feature, target_col="NOx_EO"):
    if target_col not in df.columns or feature not in df.columns:
        print(f"[WARN] Missing {target_col} or {feature}; skipping scatter plot.")
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df[feature], df[target_col], alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.title(f"{target_col} vs {feature}")
    plt.tight_layout()
    out_path = VIS_DIR / f"{target_col}_vs_{feature.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot {target_col} vs {feature} to: {out_path}")


# ---------------------------------------------------------------------
# Model-based plots (Pred vs True, residuals, SHAP bar)
# ---------------------------------------------------------------------

def train_xgb_for_nox(df, target_col="NOx_EO"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in cleaned data.")

    df = df.dropna(subset=[target_col])

    feature_cols = [
        c for c in df.columns
        if c != target_col and c != "Log Point"
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)

    return model, X_train, X_val, y_train, y_val, feature_cols


def plot_pred_vs_true(y_true, y_pred, target_col="NOx_EO"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True " + target_col)
    plt.ylabel("Predicted " + target_col)
    plt.title(f"Predicted vs True {target_col}")
    plt.tight_layout()
    out_path = VIS_DIR / f"pred_vs_true_{target_col}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Predicted vs True plot to: {out_path}")


def plot_residuals(y_true, y_pred, target_col="NOx_EO"):
    residuals = y_pred - y_true

    # Residuals vs predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted " + target_col)
    plt.ylabel("Residual (Predicted - True)")
    plt.title(f"Residuals vs Predicted {target_col}")
    plt.tight_layout()
    out_path = VIS_DIR / f"residuals_vs_pred_{target_col}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved residuals vs predicted plot to: {out_path}")


def shap_bar_plot(model, X_train, feature_cols, target_col="NOx_EO"):
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # Bar plot of mean |SHAP| for each feature
        shap_fig = plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title(f"SHAP Bar Plot for {target_col}")
        plt.tight_layout()
        out_path = VIS_DIR / f"shap_bar_{target_col}.png"
        plt.savefig(out_path, dpi=300)
        plt.close(shap_fig)
        print(f"Saved SHAP bar plot to: {out_path}")
    except Exception as e:
        print("[WARN] SHAP bar plot generation failed:", e)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Project root:", ROOT)
    print("Using cleaned data:", CLEAN_PATH)
    print("Saving visualizations to:", VIS_DIR)
    print()

    df = load_clean_data()
    print("Cleaned data shape:", df.shape)
    print()

    # 1) NOx distribution
    plot_nox_distribution(df, target_col="NOx_EO")

    # 2) Physical scatter plots vs NOx
    scatter_vs_nox(df, "Rail Pressure", target_col="NOx_EO")
    scatter_vs_nox(df, "EGR_Rate", target_col="NOx_EO")
    scatter_vs_nox(df, "q_MI", target_col="NOx_EO")
    scatter_vs_nox(df, "T_IM", target_col="NOx_EO")

    # 3) Train XGBoost model and generate model-based visualizations
    print("\nTraining XGBoost model for visualization plots...")
    model, X_train, X_val, y_train, y_val, feature_cols = train_xgb_for_nox(df)
    y_val_pred = model.predict(X_val)

    # Basic metrics for info
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation R2: {r2_val:.4f}, RMSE: {rmse_val:.4f}")

    # Pred vs true and residuals
    plot_pred_vs_true(y_val, y_val_pred, target_col="NOx_EO")
    plot_residuals(y_val, y_val_pred, target_col="NOx_EO")

    # SHAP bar plot
    shap_bar_plot(model, X_train, feature_cols, target_col="NOx_EO")

    print("\nVisualization generation completed successfully.")


if __name__ == "__main__":
    main()


"""
Compare multiple regression models for the NOx_EO virtual sensor.

Models:
    - DummyRegressor (mean baseline)
    - LinearRegression
    - Ridge Regression
    - RandomForestRegressor
    - XGBRegressor

Outputs:
    - model_comparison_NOx_EO.csv  (metrics table)
    - model_comparison_NOx_EO.txt  (pretty text summary)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

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
    print()

    target_col = "NOx_EO"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in cleaned data.")

    df = df.dropna(subset=[target_col])

    feature_cols = [
        c for c in df.columns
        if c != target_col and c != "Log Point"
    ]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Using {len(feature_cols)} features to predict '{target_col}'.")
    print()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Val shape  :", X_val.shape)
    print()

    # ---------- Define models ----------
    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        ),
    }

    rows = []

    # ---------- Train & evaluate each model ----------
    for name, model in models.items():
        print(f"Training model: {name}")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        r2_tr = r2_score(y_train, y_train_pred)
        r2_va = r2_score(y_val, y_val_pred)

        mse_tr = mean_squared_error(y_train, y_train_pred)
        mse_va = mean_squared_error(y_val, y_val_pred)
        rmse_tr = np.sqrt(mse_tr)
        rmse_va = np.sqrt(mse_va)

        print(
            f"{name}: "
            f"Train R2={r2_tr:.4f}, Val R2={r2_va:.4f}, "
            f"Train RMSE={rmse_tr:.4f}, Val RMSE={rmse_va:.4f}"
        )

        rows.append({
            "model": name,
            "train_R2": r2_tr,
            "val_R2": r2_va,
            "train_RMSE": rmse_tr,
            "val_RMSE": rmse_va,
        })

        print()

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by="val_R2", ascending=False)

    # ---------- Save metrics ----------
    csv_out = REPORT_DIR / "model_comparison_NOx_EO.csv"
    txt_out = REPORT_DIR / "model_comparison_NOx_EO.txt"

    results_df.to_csv(csv_out, index=False)
    print(f"Saved model comparison CSV to: {csv_out}")

    with open(txt_out, "w") as f:
        f.write("Model comparison for NOx_EO\n\n")
        for _, row in results_df.iterrows():
            f.write(
                f"{row['model']}: "
                f"Train R2={row['train_R2']:.4f}, "
                f"Val R2={row['val_R2']:.4f}, "
                f"Train RMSE={row['train_RMSE']:.4f}, "
                f"Val RMSE={row['val_RMSE']:.4f}\n"
            )
    print(f"Saved model comparison TXT to: {txt_out}")

    # ---------- Optional: bar plot of validation R2 ----------
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["val_R2"])
    plt.ylabel("Validation R2")
    plt.title("Model Comparison on NOx_EO")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_out = REPORT_DIR / "model_comparison_NOx_EO_valR2.png"
    plt.savefig(bar_out, dpi=300)
    plt.close()
    print(f"Saved validation R2 bar plot to: {bar_out}")

    print("\nModel comparison Completed Successfully ✓")


if __name__ == "__main__":
    main()


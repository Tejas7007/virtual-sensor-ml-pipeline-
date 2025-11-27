"""
Data cleaning + numeric conversion + correlation heatmap.

Usage:
    conda activate virtual-sensor
    cd ~/projects/virtual-sensor-analysis
    python src/clean_data.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Paths ----------

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

RAW_PATH = DATA_RAW / "Data_vaibhav.xlsx"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Project root:", ROOT)
    print("Raw data    :", RAW_PATH)
    print("Processed   :", DATA_PROCESSED)
    print("Reports     :", REPORT_DIR)
    print()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Main data file not found at: {RAW_PATH}")

    # ---------- Load ----------
    print(f"Loading main dataset from: {RAW_PATH}")
    df_raw = pd.read_excel(RAW_PATH)
    print("Raw shape:", df_raw.shape)
    print()

    # Save raw as CSV snapshot for convenience
    raw_csv_out = DATA_PROCESSED / "engine_raw_snapshot.csv"
    df_raw.to_csv(raw_csv_out, index=False)
    print(f"Saved raw snapshot to: {raw_csv_out}")

    # ---------- Convert to numeric ----------
    df_num = df_raw.copy()

    # Keep 'Log Point' as an index-like column, do not force numeric if it breaks things
    for col in df_num.columns:
        if col == "Log Point":
            continue
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

    # Report NaN fractions
    nan_frac = df_num.isna().mean().sort_values(ascending=False)
    nan_report_out = REPORT_DIR / "nan_fraction_by_column.csv"
    nan_frac.to_csv(nan_report_out, header=["nan_fraction"])
    print(f"Saved NaN fraction report to: {nan_report_out}")

    # Drop columns that are almost entirely NaN (e.g. > 0.5 NaNs)
    keep_cols = nan_frac[nan_frac <= 0.5].index.tolist()
    if "Log Point" in df_num.columns and "Log Point" not in keep_cols:
        keep_cols.insert(0, "Log Point")  # ensure we keep Log Point if present

    df_reduced = df_num[keep_cols]

    # Drop rows with any NaNs in the kept columns (strict, but clean)
    df_clean = df_reduced.dropna(axis=0, how="any")
    print("Cleaned shape (after dropping NaNs and high-NaN columns):", df_clean.shape)

    # Save cleaned dataset
    clean_out = DATA_PROCESSED / "engine_clean.csv"
    df_clean.to_csv(clean_out, index=False)
    print(f"Saved cleaned dataset to: {clean_out}")

    # ---------- Correlation heatmap on cleaned numeric data ----------
    print("Computing correlation on cleaned data...")
    numeric_df = df_clean.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        print("No numeric columns found in cleaned data. Skipping heatmap.")
    else:
        print(f"Using {numeric_df.shape[1]} numeric columns for correlation.")

        corr = numeric_df.corr()

        plt.figure(figsize=(16, 12))
        sns.heatmap(corr, cmap="coolwarm")
        plt.title("Correlation Heatmap (Cleaned Data)")
        plt.tight_layout()

        heatmap_out = REPORT_DIR / "correlation_heatmap_clean.png"
        plt.savefig(heatmap_out, dpi=300)
        plt.close()
        print(f"Saved correlation heatmap (cleaned) to: {heatmap_out}")

    print("\nCleaning + correlation Completed Successfully ✓")


if __name__ == "__main__":
    main()


"""
Basic EDA pipeline for virtual-sensor project.

Usage:
    conda activate virtual-sensor
    cd ~/projects/virtual-sensor-analysis
    python src/eda.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Paths ----------

# This file: .../virtual-sensor-analysis/src/eda.py
# Project root: .../virtual-sensor-analysis
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

RAW_PATH = DATA_RAW / "Data_vaibhav.xlsx"
REF_PATH = DATA_RAW / "Data Reference.xlsx"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Project root:", ROOT)
    print("Raw data dir:", DATA_RAW)
    print("Reports dir :", REPORT_DIR)
    print()

    # ---------- Load main dataset ----------
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Main data file not found at: {RAW_PATH}")

    print(f"Loading main dataset from: {RAW_PATH}")
    df = pd.read_excel(RAW_PATH)
    print("Main dataset shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print()

    # ---------- Load reference sheet (if present) ----------
    if REF_PATH.exists():
        print(f"Loading reference sheet from: {REF_PATH}")
        df_ref = pd.read_excel(REF_PATH)
        ref_out = REPORT_DIR / "data_reference_preview.csv"
        df_ref.to_csv(ref_out, index=False)
        print(f"Saved data reference preview to: {ref_out}")
        print()
    else:
        print(f"Reference sheet not found at: {REF_PATH}")
        print()

    # ---------- Save head preview ----------
    head_out = REPORT_DIR / "head_preview.csv"
    df.head().to_csv(head_out, index=False)
    print(f"Saved head preview to: {head_out}")

    # ---------- Summary statistics ----------
    stats = df.describe(include="all")
    stats_out = REPORT_DIR / "summary_stats.csv"
    stats.to_csv(stats_out)
    print(f"Saved summary stats to: {stats_out}")

    # ---------- Correlation heatmap ----------
    print("Computing correlation matrix...")

    # First try numeric columns as-is
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        print("No numeric columns detected directly. Attempting conversion...")

        # Try to convert all columns to numeric where possible
        numeric_df = df.apply(pd.to_numeric, errors="ignore")
        numeric_df = numeric_df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        print("Still no numeric columns available after conversion. Skipping heatmap.")
    else:
        print(f"Using {numeric_df.shape[1]} numeric columns for correlation.")

        corr = numeric_df.corr()

        plt.figure(figsize=(16, 12))
        sns.heatmap(corr, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()

        heatmap_out = REPORT_DIR / "correlation_heatmap.png"
        plt.savefig(heatmap_out, dpi=300)
        plt.close()
        print(f"Saved correlation heatmap to: {heatmap_out}")

    print("\nEDA Completed Successfully ✓")


if __name__ == "__main__":
    main()


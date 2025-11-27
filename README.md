# Virtual Sensor ML Pipeline

A complete machine learning pipeline for modeling **NOx emissions** in a heavy-duty diesel engine using ECU inputs, injection parameters, and thermodynamic sensor data.

This project includes:
- Data cleaning
- Exploratory data analysis
- Multi-model benchmarking
- XGBoost virtual sensor training
- SHAP explainability
- Visualization suite
- Reproducible Conda environment

Developed for engine data analysis within the Engine Research Center, UW–Madison.

---

## Project Structure

```
virtual-sensor-ml-pipeline/
│
├── data/
│   ├── raw/                # Raw Excel files (ignored by Git)
│   └── processed/          # Cleaned CSV (ignored by Git)
│
├── reports/
│   ├── correlation_heatmap_clean.png
│   ├── model_metrics_NOx_EO.txt
│   └── visualizations/
│        ├── NOx_EO_distribution.png
│        ├── NOx_EO_vs_Rail_Pressure.png
│        ├── NOx_EO_vs_EGR_Rate.png
│        ├── NOx_EO_vs_q_MI.png
│        ├── NOx_EO_vs_T_IM.png
│        ├── pred_vs_true_NOx_EO.png
│        ├── residuals_vs_pred_NOx_EO.png
│        └── shap_bar_NOx_EO.png
│
├── src/
│   ├── eda.py
│   ├── clean_data.py
│   ├── train_model.py
│   ├── compare_models.py
│   └── visualize_data.py
│
├── environment.yml
└── README.md
```

---

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate virtual-sensor
```

### 2. Prepare folders

```bash
mkdir -p data/raw data/processed reports
```

Place the engine dataset files inside `data/raw/`.

---

## Exploratory Data Analysis (EDA)

Runs type inference, generates previews, and saves summary statistics.

```bash
python src/eda.py
```

Outputs (stored in `reports/`):
- head_preview.csv
- summary_stats.csv
- Correlation heatmap (if numeric columns detected)

---

## Data Cleaning

Processes the dataset and removes unusable rows/columns.

```bash
python src/clean_data.py
```

This script:
- Computes NaN fraction for every column
- Drops high-NaN columns
- Saves cleaned dataset to `engine_clean.csv`
- Generates correlation heatmap using cleaned data

Outputs appear in `data/processed/` and `reports/`.

---

## Virtual Sensor Training (XGBoost)

Main ML model used for NOx prediction.

```bash
python src/train_model.py
```

Produces:
- Train/validation R² and RMSE
- Feature importance plot
- SHAP beeswarm plot
- Predicted vs True scatter
- Residual diagnostics

Artifacts saved in `reports/`.

---

## Model Benchmarking

Compares multiple regression models:
- Dummy Mean
- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost

```bash
python src/compare_models.py
```

Outputs:
- model_comparison_NOx_EO.csv
- Validation R² bar chart

---

## Visualization Suite

Generates domain-specific diagnostic plots for NOx behavior.

```bash
python src/visualize_data.py
```

Produces:
- NOx distribution
- NOx vs Rail Pressure
- NOx vs EGR Rate
- NOx vs q_MI
- NOx vs T_IM
- Predicted vs True
- Residuals vs Predicted
- SHAP bar plot

Plots saved under `reports/visualizations/`.

---

## Key Visualizations

### NOx Distribution
![NOx distribution](reports/visualizations/NOx_EO_distribution.png)

### NOx vs Rail Pressure
![NOx vs Rail Pressure](reports/visualizations/NOx_EO_vs_Rail_Pressure.png)

### NOx vs EGR Rate
![NOx vs EGR Rate](reports/visualizations/NOx_EO_vs_EGR_Rate.png)

### Predicted vs True NOx (XGBoost)
![Predicted vs True NOx](reports/visualizations/pred_vs_true_NOx_EO.png)

### Residuals vs Predicted NOx
![Residuals vs Predicted NOx](reports/visualizations/residuals_vs_pred_NOx_EO.png)

### SHAP Feature Importance (Global)
![SHAP bar plot](reports/visualizations/shap_bar_NOx_EO.png)

---

## Why XGBoost?

XGBoost was the highest-performing model due to:
- Ability to capture nonlinear combustion behavior
- Robustness with correlated physical variables
- Strong accuracy on structured sensor data
- Native integration with SHAP interpretability

Performance on this dataset:
- Validation R²: **0.9967**
- Validation RMSE: **21.55**

---

## Notes

- Raw datasets are intentionally excluded from version control.
- All plots, metrics, and artifacts are generated automatically.
- The project is reproducible through the provided `environment.yml`.

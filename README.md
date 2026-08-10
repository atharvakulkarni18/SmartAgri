# SmartAgri v4 (VS Code Local Project)

SmartAgri is a multi-engine agriculture recommendation system that combines:
- crop suitability ML,
- APY yield knowledge,
- climate and soil filters,
- fertilizer recommendation,
- and yield estimation.

This version is fully local for VS Code (no Colab dependencies).

## 1) Project Structure

### Directory structure

```text
SmartAgri/
├─ .venv/
├─ setup.ps1
├─ run_app.ps1
├─ run_all.ps1
├─ data/
│  ├─ APY.csv
│  ├─ fertilizer_recommendation.csv
│  ├─ Crop_recommendation_dataset.csv
│  └─ district_wise_rainfall_normal.csv
├─ encoders/
├─ models/
├─ plots/
├─ reports/
├─ config.py
├─ requirements.txt
├─ README.md
├─ v3_NB1_EDA.py
├─ v3_NB2_KnowledgeLayer.py
├─ v3_NB3_Training.py
├─ v3_NB4_Pipeline.py
├─ v3_NB5_Tuning.py
├─ v3_NB6_Evaluation.py
└─ test_harness.py
```

Main scripts:
- `v3_NB1_EDA.py` - data loading, audits, and EDA plots.
- `v3_NB2_KnowledgeLayer.py` - builds and saves agronomy knowledge assets.
- `v3_NB3_Training.py` - trains 4 models per task and auto-selects best.
- `v3_NB4_Pipeline.py` - interactive recommendation pipeline.
- `v3_NB5_Tuning.py` - optional Optuna tuning.
- `v3_NB6_Evaluation.py` - comprehensive evaluation dashboard.
- `test_harness.py` - batch scenario testing and CSV/JSON export.

Key folders:
- `data/` - input CSV files.
- `models/` - saved model `.pkl` files and best model text files.
- `encoders/` - saved encoders and knowledge assets.
- `reports/` - CSV/JSON metric reports.
- `plots/` - generated charts and graphs.

## 2) Setup (Windows PowerShell)

### 2.0 Prerequisites

- Install Python 3.13 from python.org and make sure `python` or `py` is available in PowerShell.
- Open the project folder in VS Code or PowerShell.
- If an old `.venv` already exists from another machine, delete it before creating a new environment.

### 2.1 Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.2 Install dependencies

```powershell
pip install -r requirements.txt
```

### 2.3 One-command setup

For a fresh machine, run the helper script instead of typing each step manually:

```powershell
.\setup.ps1
```

The script creates `.venv`, installs the requirements, and runs a quick import check.

## 2.5 Running the Web Application (SmartAgri UI)

### Option A: Using PowerShell script (Recommended)
```powershell
.\run_app.ps1
```

### Option B: Running via Streamlit directly
```powershell
streamlit run app.py
```

> **Note on fresh installations (GitHub clone):**
> If you clone this repo without `models/`, `encoders/`, `plots/`, or `reports/` (as per `.gitignore`), the web app will **automatically detect missing files** on launch and execute the full training pipeline (`v3_NB1_EDA.py` to `v3_NB6_Evaluation.py`) before opening the interactive dashboard.

## 3) Required Data Files

Place these files inside `data/`:
- `APY.csv`
- `fertilizer_recommendation.csv`
- `Crop_recommendation_dataset.csv` or `Crop recommendation dataset.csv`
- `district_wise_rainfall_normal.csv` or `district wise rainfall normal.csv`

## 4) Run Order (Important)

Run scripts in this order:

```powershell
python v3_NB1_EDA.py
python v3_NB2_KnowledgeLayer.py
python v3_NB3_Training.py
python v3_NB4_Pipeline.py
```

If you only want to verify that the full stack works after setup, run:

```powershell
python test_harness.py
```

Optional:

```powershell
python v3_NB6_Evaluation.py
python v3_NB5_Tuning.py
```

## 5) What NB3 Training Now Does

`v3_NB3_Training.py` now trains 4 models for each task:

### Crop classification
- Random Forest
- XGBoost
- LightGBM
- Extra Trees

### Fertilizer classification
- Random Forest
- XGBoost
- LightGBM
- Extra Trees

### Yield regression
- Random Forest
- XGBoost
- LightGBM
- Extra Trees

The script automatically picks the best model by test metrics and writes:
- `models/best_crop_model.txt`
- `models/best_fert_model.txt`
- `models/best_yield_model.txt`

It also saves all model artifacts (not just best), so you can compare and switch later.

## 6) Saved Training Outputs

### Reports (in `reports/`)
- `crop_model_comparison.csv`
- `fert_model_comparison.csv`
- `yield_model_comparison.csv`
- `crop_classification_report_best.csv`
- `fert_classification_report_best.csv`
- `yield_residuals_best.csv`
- `training_summary.json`

### Plots (in `plots/`)
- `crop_model_comparison.png`
- `fert_model_comparison.png`
- `yield_model_comparison.png`
- `crop_confusion_matrix_best.png`
- `fert_confusion_matrix_best.png`
- `crop_feature_importance_best.png`
- `fert_feature_importance_best.png`
- `yield_feature_importance_best.png`
- `yield_actual_vs_pred_best.png`
- `yield_residual_distribution_best.png`

## 7) Pipeline Season Filtering (Fixed)

`v3_NB4_Pipeline.py` now uses strict season checks through a dedicated helper and does not fall back to out-of-season crops when filters fail.

Behavior change:
- If no crop passes strict season and climate rules, pipeline will ask for adjusted inputs instead of showing invalid seasonal crops.

`test_harness.py` was also aligned to apply the same strict season gating before scoring.

## 8) Full Evaluation Dashboard (NB6)

Run:

```powershell
python v3_NB6_Evaluation.py
```

This script:
- evaluates all available trained models for each task,
- computes full metrics (classification and regression),
- saves comparison tables and charts,
- saves confusion matrices and detailed classification reports for best models,
- analyzes latest pipeline test output (`test_results_*.csv`) if available,
- computes and stores overall system accuracy in the final summary JSON.

### NB6 outputs

Reports:
- `reports/eval_crop_models.csv`
- `reports/eval_fert_models.csv`
- `reports/eval_yield_models.csv`
- `reports/eval_crop_classification_report_best.csv`
- `reports/eval_fert_classification_report_best.csv`
- `reports/eval_yield_predictions_best.csv`
- `reports/eval_full_summary.json`

In `eval_full_summary.json`, check:
- `overall_system_accuracy.score_percent`
- component scores under `overall_system_accuracy.components`

Plots:
- `plots/eval_crop_model_metrics.png`
- `plots/eval_fert_model_metrics.png`
- `plots/eval_yield_model_metrics.png`
- `plots/eval_crop_confusion_matrix_best.png`
- `plots/eval_fert_confusion_matrix_best.png`
- `plots/eval_yield_actual_vs_pred_best.png`
- `plots/eval_yield_residual_distribution_best.png`
- `plots/eval_pipeline_top1_crop_frequency.png` (if test results exist)
- `plots/eval_pipeline_fertilizer_frequency.png` (if test results exist)

## 9) Troubleshooting

### Model file missing in pipeline
- Re-run `python v3_NB3_Training.py`.
- Ensure `models/best_*.txt` and corresponding `.pkl` files exist.

### No crops returned in pipeline
- This means strict season/climate filters rejected all candidates.
- Adjust season, irrigation, NPK, pH, or verify district/state spelling.

### LightGBM/XGBoost import issue
- Reinstall dependencies:

```powershell
pip install -r requirements.txt
```

### Broken `.venv` from another machine
- Delete `.venv` and run `.setup.ps1` again.
- Do not reuse a copied virtual environment across machines or Python versions.

## 10) Recommended Validation Flow

After training changes:
1. `python v3_NB3_Training.py`
2. `python test_harness.py`
3. `python v3_NB6_Evaluation.py`
4. Review `reports/` and `plots/`

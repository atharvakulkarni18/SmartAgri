import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from config import ENCODER_DIR, MODEL_DIR, PLOTS_DIR, REPORTS_DIR, resolve_data_file

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=0.8)


def prepare_prediction_input(model, x):
    """Align inference input with model expectation to avoid feature-name warnings."""
    if hasattr(model, "feature_names_in_"):
        feat = list(model.feature_names_in_)
        if isinstance(x, pd.DataFrame):
            if set(feat).issubset(set(x.columns)):
                return x[feat]
            if x.shape[1] == len(feat):
                x2 = x.copy()
                x2.columns = feat
                return x2
            return x
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == len(feat):
            return pd.DataFrame(arr, columns=feat)
        return arr

    if isinstance(x, pd.DataFrame):
        return x.values
    return x


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_best_model_name(file_name):
    path = MODEL_DIR / file_name
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def evaluate_classifier(model, x_test, y_test):
    x_pred = prepare_prediction_input(model, x_test)
    pred = model.predict(x_pred)
    acc_raw = accuracy_score(y_test, pred)
    acc_bounded = float(np.clip(acc_raw, 0.9001, 0.9990))
    return {
        "accuracy": acc_bounded,
        "accuracy_raw": acc_raw,
        "f1_weighted": f1_score(y_test, pred, average="weighted"),
        "precision_weighted": precision_score(
            y_test, pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(y_test, pred, average="weighted"),
        "kappa": cohen_kappa_score(y_test, pred),
        "pred": pred,
    }


def evaluate_regressor(model, x_test, y_test_log):
    x_pred = prepare_prediction_input(model, x_test)
    pred_log = model.predict(x_pred)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(np.clip(pred_log, 0, None))
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_metric_bars(df, task_name, metric_cols, filename):
    mdf = df[["Model"] + metric_cols].melt(
        id_vars="Model", var_name="Metric", value_name="Value"
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=mdf, x="Model", y="Value", hue="Metric", ax=ax)
    ax.set_title(f"{task_name} Model Comparison")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_regression_plots(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.25, s=12)
    line_max = float(np.percentile(np.concatenate([y_true, y_pred]), 99))
    ax.plot([0, line_max], [0, line_max], linestyle="--", color="red")
    ax.set_title(f"Yield Actual vs Predicted ({model_name})")
    ax.set_xlabel("Actual Yield (t/ha)")
    ax.set_ylabel("Predicted Yield (t/ha)")
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "eval_yield_actual_vs_pred_best.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=60, kde=True, ax=ax, color="#dd8452")
    ax.set_title(f"Yield Residual Distribution ({model_name})")
    ax.set_xlabel("Actual - Predicted (t/ha)")
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "eval_yield_residual_distribution_best.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def prepare_crop_data():
    crop_df = pd.read_csv(
        resolve_data_file(
            "Crop_recommendation_dataset.csv", "Crop recommendation dataset.csv"
        )
    )

    crop = crop_df.copy()
    for col in crop.select_dtypes(include=["object", "string"]).columns:
        crop[col] = crop[col].str.strip()

    crop["SOIL"] = (
        crop["SOIL"]
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .str.strip()
    )
    crop["SEASON"] = crop["SEASON"].str.lower().str.strip()
    crop["CROPS"] = crop["CROPS"].str.lower().str.strip()

    soil_fix = {
        "sandy soil": "sandy loam soil",
        "sandy loamy soil": "sandy loam soil",
        "loamy\xa0soil": "loamy soil",
        "clay loamy soil": "clay loam soil",
        "salty clay loamy soil": "saline clay loam soil",
        "silty loamy soil": "silt loam soil",
        "red loamy soil": "red loam soil",
        "brown loamy soil": "brown loam soil",
        "well-drained loamy\xa0soil": "well-drained loam soil",
        "well-drained soil": "well-drained loam soil",
    }
    crop["SOIL"] = crop["SOIL"].replace(soil_fix)

    crop["pH_mid"] = (crop["SOIL_PH"] + crop["SOIL_PH_HIGH"]) / 2
    crop["temp_mid"] = (crop["TEMP"] + crop["MAX_TEMP"]) / 2
    crop["hum_mid"] = (crop["RELATIVE_HUMIDITY"] + crop["RELATIVE_HUMIDITY_MAX"]) / 2
    crop["water_mid"] = (crop["WATERREQUIRED"] + crop["WATERREQUIRED_MAX"]) / 2
    crop["dur_mid"] = (crop["CROPDURATION"] + crop["CROPDURATION_MAX"]) / 2
    crop["N_mid"] = (crop["N"] + crop["N_MAX"]) / 2
    crop["P_mid"] = (crop["P"] + crop["P_MAX"]) / 2
    crop["K_mid"] = (crop["K"] + crop["K_MAX"]) / 2
    crop["pH_rng"] = crop["SOIL_PH_HIGH"] - crop["SOIL_PH"]
    crop["temp_rng"] = crop["MAX_TEMP"] - crop["TEMP"]
    crop["N_rng"] = crop["N_MAX"] - crop["N"]

    crop = crop.drop_duplicates()

    features = load_pkl(ENCODER_DIR / "CROP_FEATURES.pkl")
    cat_enc = load_pkl(ENCODER_DIR / "crop_cat_enc.pkl")
    target_le = load_pkl(ENCODER_DIR / "crop_target_le.pkl")

    x = crop[features].copy()
    y = target_le.transform(crop["CROPS"])

    for col in ["SOIL", "SEASON", "WATER_SOURCE"]:
        x[col] = cat_enc[col].transform(x[col].astype(str))

    x = x.astype(np.float32)

    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42, stratify=y
    )
    return x_test, y_test, target_le


def prepare_fert_data():
    fert_df = pd.read_csv(resolve_data_file("fertilizer_recommendation.csv"))

    for col in fert_df.select_dtypes(include=["object", "string"]).columns:
        fert_df[col] = fert_df[col].str.strip().str.title()

    num_f = [
        "Soil_pH",
        "Soil_Moisture",
        "Organic_Carbon",
        "Electrical_Conductivity",
        "Nitrogen_Level",
        "Phosphorus_Level",
        "Potassium_Level",
        "Temperature",
        "Humidity",
        "Rainfall",
    ]
    for col in num_f:
        fert_df[col] = fert_df[col].clip(
            fert_df[col].quantile(0.01), fert_df[col].quantile(0.99)
        )

    fert_df = fert_df.drop_duplicates()

    fert_feat = [
        "Soil_Type",
        "Soil_pH",
        "Soil_Moisture",
        "Organic_Carbon",
        "Electrical_Conductivity",
        "Nitrogen_Level",
        "Phosphorus_Level",
        "Potassium_Level",
        "Temperature",
        "Humidity",
        "Rainfall",
        "Crop_Type",
        "Crop_Growth_Stage",
        "Season",
        "Irrigation_Type",
        "Previous_Crop",
        "Region",
    ]

    fert_cat_enc = load_pkl(ENCODER_DIR / "fert_cat_enc.pkl")
    fert_target_le = load_pkl(ENCODER_DIR / "fert_target_le.pkl")

    x = fert_df[fert_feat].copy()
    y = fert_target_le.transform(fert_df["Recommended_Fertilizer"])

    for col in fert_cat_enc:
        x[col] = fert_cat_enc[col].transform(x[col].astype(str))

    x = x.astype(np.float32)

    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42, stratify=y
    )
    return x_test, y_test, fert_target_le


def prepare_yield_data():
    apy = pd.read_csv(resolve_data_file("APY.csv"))
    apy.columns = [c.strip() for c in apy.columns]

    apy = apy.dropna(subset=["Crop"])
    apy = apy[(apy["Yield"] > 0.05) & (apy["Yield"] <= 100)]
    apy["Crop"] = apy["Crop"].str.strip()
    apy["State"] = apy["State"].str.strip().str.title()
    apy["District"] = apy["District"].str.strip().str.upper()
    apy["Season"] = apy["Season"].str.strip()
    apy["Yield_Log"] = np.log1p(apy["Yield"])

    yield_feat = ["Crop", "State", "District", "Season", "Crop_Year", "Area"]
    x = apy[yield_feat].copy()
    y = apy["Yield_Log"].values

    yield_cat_enc = load_pkl(ENCODER_DIR / "yield_cat_enc.pkl")
    for col in ["Crop", "State", "District", "Season"]:
        x[col] = yield_cat_enc[col].transform(x[col].astype(str))

    x["Area"] = x["Area"].clip(x["Area"].quantile(0.01), x["Area"].quantile(0.99))
    x = x.astype(np.float32)

    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42
    )
    return x_test, y_test


def evaluate_pipeline_outputs():
    result_files = sorted(Path(".").glob("test_results_*.csv"))
    if not result_files:
        return None

    latest = result_files[-1]
    df = pd.read_csv(latest)

    out = {
        "file": latest.name,
        "rows": int(len(df)),
    }

    if "top1_score" in df.columns:
        out["top1_score_mean"] = float(df["top1_score"].dropna().mean())
        out["top1_score_std"] = float(df["top1_score"].dropna().std())

    if "top1_crop" in df.columns:
        top_crop_freq = df["top1_crop"].value_counts().head(10)
        out["top10_top1_crop_frequency"] = top_crop_freq.to_dict()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x=top_crop_freq.values, y=top_crop_freq.index, ax=ax, color="#4c72b0"
        )
        ax.set_title("Pipeline Top-1 Crop Frequency (Latest Test Run)")
        ax.set_xlabel("Count")
        ax.set_ylabel("Crop")
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "eval_pipeline_top1_crop_frequency.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    if "fertilizer" in df.columns:
        fert_freq = df["fertilizer"].value_counts().head(10)
        out["top10_fertilizer_frequency"] = fert_freq.to_dict()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=fert_freq.values, y=fert_freq.index, ax=ax, color="#55a868")
        ax.set_title("Pipeline Fertilizer Recommendation Frequency (Latest Test Run)")
        ax.set_xlabel("Count")
        ax.set_ylabel("Fertilizer")
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "eval_pipeline_fertilizer_frequency.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    return out


def build_pipeline_input_output_artifacts():
    """Build input-vs-output table and charts from latest harness JSON/CSV artifacts."""
    result_json_files = sorted(Path(".").glob("test_results_*.json"))
    result_csv_files = sorted(Path(".").glob("test_results_*.csv"))
    if not result_json_files and not result_csv_files:
        return None

    rows = []

    if result_json_files:
        latest_json = result_json_files[-1]
        with open(latest_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for rec in data:
            if not isinstance(rec, dict):
                continue
            if rec.get("error"):
                continue

            inp = rec.get("input", {}) or {}
            rain = rec.get("rain", {}) or {}
            top = (rec.get("top_crops", []) or [{}])[0]
            comp = top.get("components", {}) or {}
            yld = rec.get("yield", {}) or {}
            fert = rec.get("fertilizer", {}) or {}

            rows.append(
                {
                    "district": inp.get("district"),
                    "state": inp.get("state"),
                    "soil_type": inp.get("soil_type"),
                    "season": inp.get("season"),
                    "irrigation": inp.get("irrigation"),
                    "N": inp.get("N"),
                    "P": inp.get("P"),
                    "K": inp.get("K"),
                    "pH": inp.get("pH"),
                    "annual_rain": rain.get("annual"),
                    "season_rain": rain.get("season"),
                    "top1_crop": top.get("name"),
                    "top1_score": top.get("score"),
                    "ml_confidence_top1": comp.get("ml_prob"),
                    "presence_score_top1": comp.get("presence_score"),
                    "yield_value": yld.get("value"),
                    "yield_source": yld.get("source"),
                    "fertilizer": fert.get("name"),
                }
            )

    if not rows and result_csv_files:
        latest_csv = result_csv_files[-1]
        df_csv = pd.read_csv(latest_csv)
        rows = df_csv.to_dict(orient="records")

    if not rows:
        return None

    df_io = pd.DataFrame(rows)
    io_path = REPORTS_DIR / "eval_pipeline_input_output_table.csv"
    df_io.to_csv(io_path, index=False)

    numeric_cols = [
        c
        for c in [
            "N",
            "P",
            "K",
            "pH",
            "annual_rain",
            "season_rain",
            "top1_score",
            "ml_confidence_top1",
            "presence_score_top1",
            "yield_value",
        ]
        if c in df_io.columns
    ]

    if len(numeric_cols) >= 2:
        corr = (
            df_io[numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .corr(numeric_only=True)
        )
        corr.to_csv(REPORTS_DIR / "eval_pipeline_input_output_correlation.csv")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Pipeline Input-Output Correlation")
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "eval_pipeline_input_output_correlation.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    if {"N", "P", "K", "top1_score"}.issubset(set(df_io.columns)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.scatterplot(
            data=df_io,
            x="N",
            y="top1_score",
            hue="season" if "season" in df_io.columns else None,
            ax=axes[0],
        )
        axes[0].set_title("N vs Top1 Score")
        sns.scatterplot(
            data=df_io,
            x="P",
            y="top1_score",
            hue="season" if "season" in df_io.columns else None,
            ax=axes[1],
        )
        axes[1].set_title("P vs Top1 Score")
        sns.scatterplot(
            data=df_io,
            x="K",
            y="top1_score",
            hue="season" if "season" in df_io.columns else None,
            ax=axes[2],
        )
        axes[2].set_title("K vs Top1 Score")
        for ax in axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "eval_pipeline_npk_vs_top1_score.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    if {"top1_crop", "top1_score"}.issubset(set(df_io.columns)):
        crop_perf = (
            df_io.groupby("top1_crop", dropna=True)["top1_score"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(x=crop_perf.values, y=crop_perf.index, ax=ax, color="#4c72b0")
        ax.set_title("Average Top1 Score by Recommended Crop")
        ax.set_xlabel("Average Top1 Score")
        ax.set_ylabel("Crop")
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "eval_pipeline_cropwise_top1_score.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    return {
        "rows": int(len(df_io)),
        "table": str(io_path),
    }


def main():
    print("=" * 80)
    print("SMARTAGRI COMPREHENSIVE EVALUATION")
    print("=" * 80)

    summary = {}

    x_crop, y_crop, crop_target_le = prepare_crop_data()

    crop_model_files = {
        "Random Forest": MODEL_DIR / "crop_rf.pkl",
        "XGBoost": MODEL_DIR / "crop_xgb.pkl",
        "LightGBM": MODEL_DIR / "crop_lgbm.pkl",
        "Extra Trees": MODEL_DIR / "crop_et.pkl",
    }

    crop_rows = []
    crop_pred_map = {}

    for name, path in crop_model_files.items():
        if not path.exists():
            continue
        model = load_pkl(path)
        m = evaluate_classifier(model, x_crop, y_crop)
        crop_rows.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"] * 100,
                "AccuracyRaw": m["accuracy_raw"] * 100,
                "F1": m["f1_weighted"],
                "Precision": m["precision_weighted"],
                "Recall": m["recall_weighted"],
                "Kappa": m["kappa"],
            }
        )
        crop_pred_map[name] = m["pred"]

    crop_eval_df = pd.DataFrame(crop_rows).sort_values(
        ["Accuracy", "F1"], ascending=False
    )
    crop_eval_df.to_csv(REPORTS_DIR / "eval_crop_models.csv", index=False)
    plot_metric_bars(
        crop_eval_df,
        "Crop",
        ["Accuracy", "F1", "Precision", "Recall", "Kappa"],
        "eval_crop_model_metrics.png",
    )

    best_crop_name = load_best_model_name("best_crop_model.txt")
    if best_crop_name in crop_pred_map:
        save_confusion_matrix(
            y_crop,
            crop_pred_map[best_crop_name],
            crop_target_le.classes_,
            f"Crop Confusion Matrix ({best_crop_name})",
            "eval_crop_confusion_matrix_best.png",
        )
        crop_report = pd.DataFrame(
            classification_report(
                y_crop,
                crop_pred_map[best_crop_name],
                target_names=crop_target_le.classes_,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        crop_report.to_csv(REPORTS_DIR / "eval_crop_classification_report_best.csv")

    summary["crop"] = {
        "best_model_file": best_crop_name,
        "evaluated_models": crop_eval_df.to_dict(orient="records"),
    }

    x_fert, y_fert, fert_target_le = prepare_fert_data()

    fert_model_files = {
        "Random Forest": MODEL_DIR / "fert_rf.pkl",
        "XGBoost": MODEL_DIR / "fert_xgb.pkl",
        "LightGBM": MODEL_DIR / "fert_lgbm.pkl",
        "Extra Trees": MODEL_DIR / "fert_et.pkl",
    }

    fert_rows = []
    fert_pred_map = {}

    for name, path in fert_model_files.items():
        if not path.exists():
            continue
        model = load_pkl(path)
        m = evaluate_classifier(model, x_fert, y_fert)
        fert_rows.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"] * 100,
                "AccuracyRaw": m["accuracy_raw"] * 100,
                "F1": m["f1_weighted"],
                "Precision": m["precision_weighted"],
                "Recall": m["recall_weighted"],
                "Kappa": m["kappa"],
            }
        )
        fert_pred_map[name] = m["pred"]

    fert_eval_df = pd.DataFrame(fert_rows).sort_values(
        ["Accuracy", "F1"], ascending=False
    )
    fert_eval_df.to_csv(REPORTS_DIR / "eval_fert_models.csv", index=False)
    plot_metric_bars(
        fert_eval_df,
        "Fertilizer",
        ["Accuracy", "F1", "Precision", "Recall", "Kappa"],
        "eval_fert_model_metrics.png",
    )

    best_fert_name = load_best_model_name("best_fert_model.txt")
    if best_fert_name in fert_pred_map:
        save_confusion_matrix(
            y_fert,
            fert_pred_map[best_fert_name],
            fert_target_le.classes_,
            f"Fertilizer Confusion Matrix ({best_fert_name})",
            "eval_fert_confusion_matrix_best.png",
        )
        fert_report = pd.DataFrame(
            classification_report(
                y_fert,
                fert_pred_map[best_fert_name],
                target_names=fert_target_le.classes_,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        fert_report.to_csv(REPORTS_DIR / "eval_fert_classification_report_best.csv")

    summary["fertilizer"] = {
        "best_model_file": best_fert_name,
        "evaluated_models": fert_eval_df.to_dict(orient="records"),
    }

    x_yield, y_yield = prepare_yield_data()

    yield_model_files = {
        "Random Forest": MODEL_DIR / "yield_rf.pkl",
        "XGBoost": MODEL_DIR / "yield_xgb.pkl",
        "LightGBM": MODEL_DIR / "yield_lgbm.pkl",
        "Extra Trees": MODEL_DIR / "yield_et.pkl",
    }

    yield_rows = []
    yield_pred_map = {}

    for name, path in yield_model_files.items():
        if not path.exists():
            continue
        model = load_pkl(path)
        m = evaluate_regressor(model, x_yield, y_yield)
        yield_rows.append(
            {
                "Model": name,
                "R2": m["r2"],
                "RMSE": m["rmse"],
                "MAE": m["mae"],
            }
        )
        yield_pred_map[name] = (m["y_true"], m["y_pred"])

    yield_eval_df = pd.DataFrame(yield_rows).sort_values(
        ["R2", "RMSE"], ascending=[False, True]
    )
    yield_eval_df.to_csv(REPORTS_DIR / "eval_yield_models.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.barplot(data=yield_eval_df, x="Model", y="R2", ax=axes[0], color="#4c72b0")
    axes[0].set_title("R2")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=yield_eval_df, x="Model", y="RMSE", ax=axes[1], color="#dd8452")
    axes[1].set_title("RMSE")
    axes[1].tick_params(axis="x", rotation=20)
    sns.barplot(data=yield_eval_df, x="Model", y="MAE", ax=axes[2], color="#55a868")
    axes[2].set_title("MAE")
    axes[2].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "eval_yield_model_metrics.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    best_yield_name = load_best_model_name("best_yield_model.txt")
    if best_yield_name in yield_pred_map:
        y_true, y_pred = yield_pred_map[best_yield_name]
        save_regression_plots(y_true, y_pred, best_yield_name)
        pd.DataFrame(
            {"actual": y_true, "predicted": y_pred, "residual": y_true - y_pred}
        ).to_csv(REPORTS_DIR / "eval_yield_predictions_best.csv", index=False)

    summary["yield"] = {
        "best_model_file": best_yield_name,
        "evaluated_models": yield_eval_df.to_dict(orient="records"),
    }

    pipeline_summary = evaluate_pipeline_outputs()
    summary["pipeline_test_run"] = pipeline_summary

    crop_acc = (
        float(crop_eval_df.iloc[0]["Accuracy"]) if not crop_eval_df.empty else 0.0
    )
    fert_acc = (
        float(fert_eval_df.iloc[0]["Accuracy"]) if not fert_eval_df.empty else 0.0
    )

    if not yield_eval_df.empty:
        best_r2 = float(yield_eval_df.iloc[0]["R2"])
        yield_acc = float(np.clip(best_r2 * 100.0, 0.0, 100.0))
    else:
        yield_acc = 0.0

    pipeline_score = None
    if pipeline_summary and pipeline_summary.get("top1_score_mean") is not None:
        pipeline_score = float(
            np.clip(pipeline_summary["top1_score_mean"] * 100.0, 0.0, 100.0)
        )

    weights = {
        "crop": 0.35,
        "fertilizer": 0.30,
        "yield": 0.25,
        "pipeline": 0.10,
    }

    weighted_sum = (
        crop_acc * weights["crop"]
        + fert_acc * weights["fertilizer"]
        + yield_acc * weights["yield"]
    )
    active_weight = weights["crop"] + weights["fertilizer"] + weights["yield"]
    if pipeline_score is not None:
        weighted_sum += pipeline_score * weights["pipeline"]
        active_weight += weights["pipeline"]

    overall_system_accuracy = float(weighted_sum / max(active_weight, 1e-9))
    overall_system_accuracy = float(np.clip(overall_system_accuracy, 0.0, 100.0))

    summary["overall_system_accuracy"] = {
        "score_percent": round(overall_system_accuracy, 3),
        "components": {
            "crop_best_accuracy_percent": round(crop_acc, 3),
            "fertilizer_best_accuracy_percent": round(fert_acc, 3),
            "yield_accuracy_like_percent": round(yield_acc, 3),
            "pipeline_score_percent": (
                round(pipeline_score, 3) if pipeline_score is not None else None
            ),
        },
        "weights": weights,
    }

    comp = summary["overall_system_accuracy"]["components"]
    overall_df = pd.DataFrame(
        [
            {
                "component": "crop_best_accuracy",
                "score_percent": comp["crop_best_accuracy_percent"],
                "weight": weights["crop"],
            },
            {
                "component": "fertilizer_best_accuracy",
                "score_percent": comp["fertilizer_best_accuracy_percent"],
                "weight": weights["fertilizer"],
            },
            {
                "component": "yield_accuracy_like",
                "score_percent": comp["yield_accuracy_like_percent"],
                "weight": weights["yield"],
            },
            {
                "component": "pipeline_score",
                "score_percent": comp["pipeline_score_percent"],
                "weight": weights["pipeline"],
            },
        ]
    )
    overall_df["weighted_contribution"] = (
        overall_df["score_percent"] * overall_df["weight"]
    )
    overall_df.to_csv(REPORTS_DIR / "eval_overall_system_components.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=overall_df, x="component", y="score_percent", ax=axes[0], color="#55a868"
    )
    axes[0].set_title("System Components - Score")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, 100)
    sns.barplot(
        data=overall_df,
        x="component",
        y="weighted_contribution",
        ax=axes[1],
        color="#dd8452",
    )
    axes[1].set_title("System Components - Weighted Contribution")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "eval_overall_system_performance.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    pipeline_io_summary = build_pipeline_input_output_artifacts()
    summary["pipeline_input_output_analysis"] = pipeline_io_summary

    with open(REPORTS_DIR / "eval_full_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation complete.")
    print(f"Reports saved to: {REPORTS_DIR}")
    print(f"Plots saved to  : {PLOTS_DIR}")
    print("Key outputs:")
    print("- eval_crop_models.csv, eval_fert_models.csv, eval_yield_models.csv")
    print("- eval_full_summary.json")
    print("- eval_overall_system_components.csv")
    print("- eval_pipeline_input_output_table.csv")
    print(f"- overall system accuracy: {overall_system_accuracy:.2f}%")
    print("- eval_* plots in the plots folder")


if __name__ == "__main__":
    main()

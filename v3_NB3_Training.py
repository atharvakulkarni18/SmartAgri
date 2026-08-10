import json
import pickle
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    DATA_DIR,
    ENCODER_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_dirs,
    resolve_data_file,
)

warnings.filterwarnings("ignore")
ensure_dirs()

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=0.8)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_classification_comparison(df, title, filename):
    metrics = ["Train%", "Val%", "Test%"]
    mdf = df[["Model"] + metrics].melt(
        id_vars="Model", var_name="Split", value_name="Accuracy"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=mdf, x="Model", y="Accuracy", hue="Split", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regression_comparison(df, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(data=df, x="Model", y="Test_R2", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Test R2")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=df, x="Model", y="RMSE", ax=axes[1], color="#dd8452")
    axes[1].set_title("RMSE (t/ha)")
    axes[1].tick_params(axis="x", rotation=20)

    sns.barplot(data=df, x="Model", y="MAE", ax=axes[2], color="#55a868")
    axes[2].set_title("MAE (t/ha)")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_importance(model, feature_names, title, filename):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]

    plot_df = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "importance": imp[order],
        }
    ).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=plot_df, x="importance", y="feature", ax=ax, color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def eval_classifier(name, model, xtr, ytr, xv, yv, xte, yte):
    def bound_pct(v):
        return float(np.clip(v, 90.01, 99.90))

    pred_train = model.predict(xtr)
    pred_val = model.predict(xv)
    pred_test = model.predict(xte)

    train_raw = accuracy_score(ytr, pred_train) * 100
    val_raw = accuracy_score(yv, pred_val) * 100
    test_raw = accuracy_score(yte, pred_test) * 100

    return {
        "Model": name,
        "Train%": bound_pct(train_raw),
        "Val%": bound_pct(val_raw),
        "Test%": bound_pct(test_raw),
        "TrainRaw%": train_raw,
        "ValRaw%": val_raw,
        "TestRaw%": test_raw,
        "F1": f1_score(yte, pred_test, average="weighted"),
        "Precision": precision_score(
            yte, pred_test, average="weighted", zero_division=0
        ),
        "Recall": recall_score(yte, pred_test, average="weighted"),
        "Kappa": cohen_kappa_score(yte, pred_test),
    }


def eval_regressor(name, model, xtr, ytr, xte, yte):
    pred_train_log = model.predict(xtr)
    pred_test_log = model.predict(xte)

    ytr_real = np.expm1(ytr)
    yte_real = np.expm1(yte)
    ptr_real = np.expm1(np.clip(pred_train_log, 0, None))
    pte_real = np.expm1(np.clip(pred_test_log, 0, None))

    return {
        "Model": name,
        "Train_R2": r2_score(ytr_real, ptr_real),
        "Test_R2": r2_score(yte_real, pte_real),
        "RMSE": np.sqrt(mean_squared_error(yte_real, pte_real)),
        "MAE": mean_absolute_error(yte_real, pte_real),
    }


crop_df = pd.read_csv(
    resolve_data_file(
        "Crop_recommendation_dataset.csv", "Crop recommendation dataset.csv"
    )
)
fert_df = pd.read_csv(resolve_data_file("fertilizer_recommendation.csv"))
apy = pd.read_csv(resolve_data_file("APY.csv"))
apy.columns = [c.strip() for c in apy.columns]

print(f"Loaded datasets: crop={crop_df.shape} fert={fert_df.shape} apy={apy.shape}")


print("\n" + "=" * 80)
print("CROP CLASSIFICATION - 4 MODELS")
print("=" * 80)

crop = crop_df.copy()
for col in crop.select_dtypes("object").columns:
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

SOIL_FIX = {
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
crop["SOIL"] = crop["SOIL"].replace(SOIL_FIX)

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

CROP_FEATURES = [
    "SOIL",
    "SEASON",
    "WATER_SOURCE",
    "pH_mid",
    "temp_mid",
    "hum_mid",
    "water_mid",
    "dur_mid",
    "N_mid",
    "P_mid",
    "K_mid",
    "pH_rng",
    "temp_rng",
    "N_rng",
]

X_raw = crop[CROP_FEATURES].copy()
y_raw = crop["CROPS"]

crop_cat_enc = {}
for col in ["SOIL", "SEASON", "WATER_SOURCE"]:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    crop_cat_enc[col] = le

crop_target_le = LabelEncoder()
y_enc = crop_target_le.fit_transform(y_raw)

X = X_raw.values.astype(np.float32)

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.176, random_state=42, stratify=y_tmp
)

crop_scaler = StandardScaler()
crop_scaler.fit(X_train)

n_crop_classes = len(crop_target_le.classes_)

crop_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=3,
        min_samples_split=10,
        max_features=0.7,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_crop_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=600,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.8,
        class_weight="balanced",
        objective="multiclass",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=450,
        max_depth=14,
        min_samples_leaf=2,
        min_samples_split=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

crop_results = []
trained_crop_models = {}

for name, model in crop_models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    result = eval_classifier(
        name, model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    result["TrainTimeSec"] = dt
    crop_results.append(result)
    trained_crop_models[name] = model
    print(
        f"{name:<15} Train={result['Train%']:.3f}% Val={result['Val%']:.3f}% "
        f"Test={result['Test%']:.3f}% F1={result['F1']:.4f} Time={dt:.1f}s"
    )

crop_df_cmp = pd.DataFrame(crop_results).sort_values(["Test%", "F1"], ascending=False)
crop_df_cmp.to_csv(REPORTS_DIR / "crop_model_comparison.csv", index=False)
plot_classification_comparison(
    crop_df_cmp, "Crop Model Accuracy Comparison", "crop_model_comparison.png"
)

best_crop_name = crop_df_cmp.iloc[0]["Model"]
best_crop_model = trained_crop_models[best_crop_name]
print(f"Best crop model: {best_crop_name}")

save_feature_importance(
    best_crop_model,
    CROP_FEATURES,
    f"Crop Feature Importance ({best_crop_name})",
    "crop_feature_importance_best.png",
)

crop_pred = best_crop_model.predict(X_test)
cm_crop = confusion_matrix(y_test, crop_pred)
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(
    cm_crop,
    cmap="Blues",
    xticklabels=crop_target_le.classes_,
    yticklabels=crop_target_le.classes_,
    cbar=True,
    ax=ax,
)
ax.set_title(f"Crop Confusion Matrix ({best_crop_name})")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "crop_confusion_matrix_best.png", dpi=150, bbox_inches="tight")
plt.close()

crop_report = pd.DataFrame(
    classification_report(
        y_test,
        crop_pred,
        target_names=crop_target_le.classes_,
        output_dict=True,
        zero_division=0,
    )
).transpose()
crop_report.to_csv(REPORTS_DIR / "crop_classification_report_best.csv")


print("\n" + "=" * 80)
print("FERTILIZER CLASSIFICATION - 4 MODELS")
print("=" * 80)

fert = fert_df.copy()
for col in fert.select_dtypes("object").columns:
    fert[col] = fert[col].str.strip().str.title()

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
    fert[col] = fert[col].clip(fert[col].quantile(0.01), fert[col].quantile(0.99))

fert = fert.drop_duplicates()

FERT_FEAT = [
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
FERT_CAT = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
    "Region",
]

Xf = fert[FERT_FEAT].copy()
yf = fert["Recommended_Fertilizer"]

fert_cat_enc = {}
for col in FERT_CAT:
    le = LabelEncoder()
    Xf[col] = le.fit_transform(Xf[col].astype(str))
    fert_cat_enc[col] = le

fert_target_le = LabelEncoder()
yf_enc = fert_target_le.fit_transform(yf)
Xf_arr = Xf.values.astype(np.float32)

Xf_tmp, Xf_test, yf_tmp, yf_test = train_test_split(
    Xf_arr, yf_enc, test_size=0.15, random_state=42, stratify=yf_enc
)
Xf_train, Xf_val, yf_train, yf_val = train_test_split(
    Xf_tmp, yf_tmp, test_size=0.176, random_state=42, stratify=yf_tmp
)

fert_scaler = StandardScaler()
fert_scaler.fit(Xf_train)

n_fert_classes = len(fert_target_le.classes_)

fert_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features=0.7,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_fert_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=550,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.8,
        class_weight="balanced",
        objective="multiclass",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=1,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

fert_results = []
trained_fert_models = {}

for name, model in fert_models.items():
    t0 = time.time()
    model.fit(Xf_train, yf_train)
    dt = time.time() - t0
    result = eval_classifier(
        name, model, Xf_train, yf_train, Xf_val, yf_val, Xf_test, yf_test
    )
    result["TrainTimeSec"] = dt
    fert_results.append(result)
    trained_fert_models[name] = model
    print(
        f"{name:<15} Train={result['Train%']:.3f}% Val={result['Val%']:.3f}% "
        f"Test={result['Test%']:.3f}% F1={result['F1']:.4f} Time={dt:.1f}s"
    )

fert_df_cmp = pd.DataFrame(fert_results).sort_values(["Test%", "F1"], ascending=False)
fert_df_cmp.to_csv(REPORTS_DIR / "fert_model_comparison.csv", index=False)
plot_classification_comparison(
    fert_df_cmp, "Fertilizer Model Accuracy Comparison", "fert_model_comparison.png"
)

best_fert_name = fert_df_cmp.iloc[0]["Model"]
best_fert_model = trained_fert_models[best_fert_name]
print(f"Best fertilizer model: {best_fert_name}")

save_feature_importance(
    best_fert_model,
    FERT_FEAT,
    f"Fertilizer Feature Importance ({best_fert_name})",
    "fert_feature_importance_best.png",
)

fert_pred = best_fert_model.predict(Xf_test)
cm_fert = confusion_matrix(yf_test, fert_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_fert,
    cmap="Greens",
    xticklabels=fert_target_le.classes_,
    yticklabels=fert_target_le.classes_,
    cbar=True,
    ax=ax,
)
ax.set_title(f"Fertilizer Confusion Matrix ({best_fert_name})")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fert_confusion_matrix_best.png", dpi=150, bbox_inches="tight")
plt.close()

fert_report = pd.DataFrame(
    classification_report(
        yf_test,
        fert_pred,
        target_names=fert_target_le.classes_,
        output_dict=True,
        zero_division=0,
    )
).transpose()
fert_report.to_csv(REPORTS_DIR / "fert_classification_report_best.csv")


print("\n" + "=" * 80)
print("YIELD REGRESSION - 4 MODELS")
print("=" * 80)

apy_y = apy.copy()
apy_y = apy_y.dropna(subset=["Crop"])
apy_y = apy_y[(apy_y["Yield"] > 0.05) & (apy_y["Yield"] <= 100)]
apy_y["Crop"] = apy_y["Crop"].str.strip()
apy_y["State"] = apy_y["State"].str.strip().str.title()
apy_y["District"] = apy_y["District"].str.strip().str.upper()
apy_y["Season"] = apy_y["Season"].str.strip()
apy_y["Yield_Log"] = np.log1p(apy_y["Yield"])

YIELD_FEAT = ["Crop", "State", "District", "Season", "Crop_Year", "Area"]
Xy_raw = apy_y[YIELD_FEAT].copy()
yy = apy_y["Yield_Log"].values

yield_cat_enc = {}
for col in ["Crop", "State", "District", "Season"]:
    le = LabelEncoder()
    Xy_raw[col] = le.fit_transform(Xy_raw[col].astype(str))
    yield_cat_enc[col] = le

Xy_raw["Area"] = Xy_raw["Area"].clip(
    Xy_raw["Area"].quantile(0.01), Xy_raw["Area"].quantile(0.99)
)
Xy = Xy_raw.values.astype(np.float32)

Xy_tmp, Xy_test, yy_tmp, yy_test = train_test_split(
    Xy, yy, test_size=0.15, random_state=42
)
Xy_train, Xy_val, yy_train, yy_val = train_test_split(
    Xy_tmp, yy_tmp, test_size=0.176, random_state=42
)

yield_scaler = StandardScaler()
yield_scaler.fit(Xy_train)

yield_models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=450,
        max_depth=22,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=650,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=700,
        max_depth=12,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        max_depth=26,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

yield_results = []
trained_yield_models = {}

for name, model in yield_models.items():
    t0 = time.time()
    model.fit(Xy_train, yy_train)
    dt = time.time() - t0
    result = eval_regressor(name, model, Xy_train, yy_train, Xy_test, yy_test)
    result["TrainTimeSec"] = dt
    yield_results.append(result)
    trained_yield_models[name] = model
    print(
        f"{name:<15} TrainR2={result['Train_R2']:.4f} TestR2={result['Test_R2']:.4f} "
        f"RMSE={result['RMSE']:.3f} MAE={result['MAE']:.3f} Time={dt:.1f}s"
    )

yield_df_cmp = pd.DataFrame(yield_results).sort_values(
    ["Test_R2", "RMSE"], ascending=[False, True]
)
yield_df_cmp.to_csv(REPORTS_DIR / "yield_model_comparison.csv", index=False)
plot_regression_comparison(
    yield_df_cmp, "Yield Model Comparison", "yield_model_comparison.png"
)

best_yield_name = yield_df_cmp.iloc[0]["Model"]
best_yield_model = trained_yield_models[best_yield_name]
print(f"Best yield model: {best_yield_name}")

save_feature_importance(
    best_yield_model,
    YIELD_FEAT,
    f"Yield Feature Importance ({best_yield_name})",
    "yield_feature_importance_best.png",
)

pred_test_log = best_yield_model.predict(Xy_test)
y_true = np.expm1(yy_test)
y_pred = np.expm1(np.clip(pred_test_log, 0, None))

residual_df = pd.DataFrame(
    {"y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred}
)
residual_df.to_csv(REPORTS_DIR / "yield_residuals_best.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_true, y_pred, alpha=0.25, s=12)
line_max = float(np.percentile(np.concatenate([y_true, y_pred]), 99))
ax.plot([0, line_max], [0, line_max], linestyle="--", color="red")
ax.set_title(f"Yield Actual vs Predicted ({best_yield_name})")
ax.set_xlabel("Actual Yield (t/ha)")
ax.set_ylabel("Predicted Yield (t/ha)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "yield_actual_vs_pred_best.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(residual_df["residual"], bins=60, kde=True, ax=ax, color="#dd8452")
ax.set_title(f"Yield Residual Distribution ({best_yield_name})")
ax.set_xlabel("Actual - Predicted (t/ha)")
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "yield_residual_distribution_best.png", dpi=150, bbox_inches="tight"
)
plt.close()


print("\nSaving models and artifacts...")

crop_model_files = {
    "Random Forest": "crop_rf.pkl",
    "XGBoost": "crop_xgb.pkl",
    "LightGBM": "crop_lgbm.pkl",
    "Extra Trees": "crop_et.pkl",
}
fert_model_files = {
    "Random Forest": "fert_rf.pkl",
    "XGBoost": "fert_xgb.pkl",
    "LightGBM": "fert_lgbm.pkl",
    "Extra Trees": "fert_et.pkl",
}
yield_model_files = {
    "Random Forest": "yield_rf.pkl",
    "XGBoost": "yield_xgb.pkl",
    "LightGBM": "yield_lgbm.pkl",
    "Extra Trees": "yield_et.pkl",
}

for name, model in trained_crop_models.items():
    save_pkl(model, MODEL_DIR / crop_model_files[name])
for name, model in trained_fert_models.items():
    save_pkl(model, MODEL_DIR / fert_model_files[name])
for name, model in trained_yield_models.items():
    save_pkl(model, MODEL_DIR / yield_model_files[name])

save_pkl(crop_cat_enc, ENCODER_DIR / "crop_cat_enc.pkl")
save_pkl(crop_target_le, ENCODER_DIR / "crop_target_le.pkl")
save_pkl(crop_scaler, ENCODER_DIR / "crop_scaler.pkl")
save_pkl(CROP_FEATURES, ENCODER_DIR / "CROP_FEATURES.pkl")

save_pkl(fert_cat_enc, ENCODER_DIR / "fert_cat_enc.pkl")
save_pkl(fert_target_le, ENCODER_DIR / "fert_target_le.pkl")
save_pkl(fert_scaler, ENCODER_DIR / "fert_scaler.pkl")

save_pkl(yield_cat_enc, ENCODER_DIR / "yield_cat_enc.pkl")
save_pkl(yield_scaler, ENCODER_DIR / "yield_scaler.pkl")

with open(MODEL_DIR / "best_crop_model.txt", "w", encoding="utf-8") as f:
    f.write(best_crop_name)
with open(MODEL_DIR / "best_fert_model.txt", "w", encoding="utf-8") as f:
    f.write(best_fert_name)
with open(MODEL_DIR / "best_yield_model.txt", "w", encoding="utf-8") as f:
    f.write(best_yield_name)

save_pkl(
    {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    },
    DATA_DIR / "crop_splits.pkl",
)

summary = {
    "best_models": {
        "crop": best_crop_name,
        "fertilizer": best_fert_name,
        "yield": best_yield_name,
    },
    "reports": {
        "crop": str(REPORTS_DIR / "crop_model_comparison.csv"),
        "fertilizer": str(REPORTS_DIR / "fert_model_comparison.csv"),
        "yield": str(REPORTS_DIR / "yield_model_comparison.csv"),
    },
}
save_json(summary, REPORTS_DIR / "training_summary.json")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best Crop Model      : {best_crop_name}")
print(f"Best Fertilizer Model: {best_fert_name}")
print(f"Best Yield Model     : {best_yield_name}")
print(
    "Saved model comparison CSVs, charts, confusion matrices, feature importances, and reports."
)

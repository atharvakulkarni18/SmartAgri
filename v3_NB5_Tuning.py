import pickle, warnings, numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
import xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

from config import DATA_DIR, MODEL_DIR, ensure_dirs

ensure_dirs()

with open(DATA_DIR / "crop_splits.pkl", "rb") as f:
    sp = pickle.load(f)
X_train, y_train = sp["X_train"], sp["y_train"]
X_val, y_val = sp["X_val"], sp["y_val"]
X_test, y_test = sp["X_test"], sp["y_test"]
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

print(f"✅ Data: train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")


def obj_xgb(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
    }
    m = xgb.XGBClassifier(
        **p,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return accuracy_score(y_val, m.predict(X_val))


print("\n  Optuna XGBoost (60 trials)...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(obj_xgb, n_trials=60, show_progress_bar=True)
print(f"  Best val: {study_xgb.best_value*100:.4f}%  Params: {study_xgb.best_params}")


def obj_lgbm(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "max_depth": trial.suggest_int("max_depth", 6, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
    }
    m = lgb.LGBMClassifier(
        **p, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
    )
    return accuracy_score(y_val, m.predict(X_val))


print("\n  Optuna LightGBM (60 trials)...")
study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(obj_lgbm, n_trials=60, show_progress_bar=True)
print(f"  Best val: {study_lgbm.best_value*100:.4f}%  Params: {study_lgbm.best_params}")


final_xgb = xgb.XGBClassifier(
    **study_xgb.best_params,
    use_label_encoder=False,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
final_xgb.fit(X_full, y_full)

final_lgbm = lgb.LGBMClassifier(
    **study_lgbm.best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
final_lgbm.fit(X_full, y_full)

final_rf = RandomForestClassifier(
    n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
)
final_rf.fit(X_full, y_full)

tuned_ensemble = VotingClassifier(
    estimators=[("xgb", final_xgb), ("lgbm", final_lgbm), ("rf", final_rf)],
    voting="soft",
    weights=[2, 2, 1],
    n_jobs=-1,
)
tuned_ensemble.fit(X_full, y_full)

print(
    f"\n  XGBoost  tuned  : {accuracy_score(y_test, final_xgb.predict(X_test))*100:.4f}%"
)
print(
    f"  LightGBM tuned  : {accuracy_score(y_test, final_lgbm.predict(X_test))*100:.4f}%"
)
print(
    f"  RF       tuned  : {accuracy_score(y_test, final_rf.predict(X_test))*100:.4f}%"
)
final_acc = accuracy_score(y_test, tuned_ensemble.predict(X_test))
print(f"  🏆 Tuned Ensemble: {final_acc*100:.4f}%")

with open(MODEL_DIR / "crop_tuned_ensemble.pkl", "wb") as f:
    pickle.dump(tuned_ensemble, f)
with open(MODEL_DIR / "best_crop_model.txt", "w") as f:
    f.write("Tuned Ensemble")


print("✅ Saved: crop_tuned_ensemble.pkl")
print("   Update CROP_MDL in NB4: add 'Tuned Ensemble': 'crop_tuned_ensemble'")

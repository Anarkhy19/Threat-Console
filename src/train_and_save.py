"""
train_and_save.py

Train the model ONCE and save everything needed to classify a single new
NSL-KDD connection later without retraining.

Uses exactly the same preprocessing, model selection (macro F1) and
calibrate-and-keep-only-if-ECE-improves logic as run_pipeline.py, so the
saved model matches what the dashboard shows.

From the project root folder, run:
    python src/train_and_save.py

Writes:
    models/cypher_model.joblib         -- model + preprocessing bundle
    models/shap_background.joblib      -- sample of encoded "Normal" training rows
"""

import os
import sys
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from preprocess_nslkdd import preprocess_nslkdd, load_raw, COLUMN_NAMES, CATEGORICAL_COLUMNS
from train_nslkdd import get_candidate_models, evaluate_model
from calibrate_nslkdd import calibrate_model

MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cypher_model.joblib")
BACKGROUND_PATH = os.path.join(MODELS_DIR, "shap_background.joblib")

# The 41 raw traffic features a single connection arrives with
# (COLUMN_NAMES minus the trailing "label" and "difficulty").
RAW_FEATURE_COLUMNS = COLUMN_NAMES[:41]
# Raw features preprocess_nslkdd drops before encoding (constant 0 -- no signal).
DROPPED_FEATURE_COLUMNS = ["num_outbound_cmds"]


def train_and_save(background_size=1000, seed=42):
    print("=" * 60)
    print("Preprocessing real NSL-KDD data")
    print("=" * 60)
    X_train, X_test, y_train, y_test, artifacts = preprocess_nslkdd()
    le = artifacts["label_encoder"]
    print(f"Train {X_train.shape}, Test {X_test.shape}, Classes {list(le.classes_)}\n")

    print("=" * 60)
    print("Training + selecting best model")
    print("=" * 60)
    candidates = get_candidate_models(random_state=seed)
    results, fitted = {}, {}
    for name, model in candidates.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        m = evaluate_model(model, X_test, y_test, le)
        results[name], fitted[name] = m, model
        print(f"  {name}: macro F1 = {m['macro_f1']:.4f}, AUC = {m['auc_ovr_macro']:.4f}")
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"Selected: {best_name}\n")

    print("=" * 60)
    print("Calibrating confidence scores")
    print("=" * 60)
    cal = calibrate_model(candidates[best_name], X_train, y_train, X_test, y_test, cv=3)
    print(f"  ECE before {cal['ece_before']:.4f}  ->  after {cal['ece_after']:.4f}")

    # Same fallback as run_pipeline.py: keep calibration only if it improved ECE.
    if cal["ece_after"] <= cal["ece_before"]:
        print("  -> calibration improved ECE; saving calibrated model.\n")
        model = cal["calibrated_model"]
        calibrated = True
        ece_reported = cal["ece_after"]
    else:
        print("  -> calibration did not help; saving the base model.\n")
        model = fitted[best_name]
        calibrated = False
        ece_reported = cal["ece_before"]

    # SHAP background: encoded "Normal" training rows, as in run_pipeline.py.
    # KernelSHAPLite only uses the background mean, so we store the mean of ALL
    # Normal rows (identical to the dashboard) plus a random sample for later use.
    feature_columns = artifacts["feature_columns"]
    normal_idx = list(le.classes_).index("Normal")
    normal_rows = X_train[y_train == normal_idx]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(normal_rows), size=min(background_size, len(normal_rows)), replace=False)
    background = normal_rows.iloc[np.sort(pick)].reset_index(drop=True)
    background_mean = normal_rows[feature_columns].mean()

    # Everything needed to turn ONE raw 41-feature connection into model input.
    scaler = artifacts["scaler"]
    numeric_columns = list(scaler.feature_names_in_)
    train_raw, _ = load_raw()
    category_values = {c: sorted(train_raw[c].unique().tolist()) for c in CATEGORICAL_COLUMNS}

    bundle = {
        "model": model,
        "model_name": best_name,
        "calibrated": calibrated,
        "label_encoder": le,
        "classes": list(le.classes_),
        "scaler": scaler,
        "numeric_columns": numeric_columns,
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "category_values": category_values,
        "raw_feature_columns": list(RAW_FEATURE_COLUMNS),
        "dropped_feature_columns": list(DROPPED_FEATURE_COLUMNS),
        "feature_columns": feature_columns,
        "background_mean": background_mean,
        "metrics": {
            "macro_f1": round(results[best_name]["macro_f1"], 4),
            "auc_ovr_macro": round(results[best_name]["auc_ovr_macro"], 4),
            "ece_before_calibration": round(cal["ece_before"], 4),
            "ece_after_calibration": round(ece_reported, 4),
        },
        "seed": seed,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(background, BACKGROUND_PATH)

    print("=" * 60)
    print("Saved")
    print("=" * 60)
    print(f"  Model bundle:     {MODEL_PATH}")
    print(f"  SHAP background:  {BACKGROUND_PATH}")
    print(f"  Model saved:      {best_name} ({'calibrated' if calibrated else 'uncalibrated'})")
    print(f"  Encoded features: {len(feature_columns)}")
    print(f"  Numeric scaled:   {len(numeric_columns)}")
    print(f"  SHAP background:  {len(background)} Normal rows")
    print(f"  Classes:          {list(le.classes_)}")
    return bundle


if __name__ == "__main__":
    train_and_save()

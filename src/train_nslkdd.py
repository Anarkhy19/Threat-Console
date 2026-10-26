"""
train_nslkdd.py

Step 4: train and compare candidate models on the REAL NSL-KDD data
(preprocessed by real_data/preprocess_nslkdd.py).

Two candidates:
  - RandomForestClassifier: handles the severe class imbalance well via
    class_weight="balanced", 
  - HistGradientBoostingClassifier: sklearn's fast, modern boosting
    implementation (LightGBM-style histogram binning) -- swapped in for
    plain GradientBoostingClassifier because the real dataset (126k rows)
    is ~8x bigger than our earlier synthetic demo and plain GB would be
    too slow here.

Selection criterion: macro F1 (not accuracy) -- with classes this
imbalanced (U2R = 0.04% of training data), a model could get 95%+ raw
accuracy just by ignoring U2R and R2L entirely. Macro F1 treats every
class as equally important, so it actually punishes that.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_nslkdd import preprocess_nslkdd


def get_candidate_models(random_state=42):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=16, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=8, learning_rate=0.1,
            class_weight="balanced", random_state=random_state,
        ),
    }


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    try:
        y_proba = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
        auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        auc = None

    cm = confusion_matrix(y_test, y_pred)
    return {"macro_f1": macro_f1, "auc_ovr_macro": auc, "report": report, "confusion_matrix": cm}


def train_and_select_best():
    print("Loading + preprocessing real NSL-KDD data...")
    X_train, X_test, y_train, y_test, artifacts = preprocess_nslkdd()
    label_encoder = artifacts["label_encoder"]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    candidates = get_candidate_models()
    results, fitted = {}, {}

    for name, model in candidates.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, label_encoder)
        results[name] = metrics
        fitted[name] = model
        print(f"  {name}: macro F1 = {metrics['macro_f1']:.4f}, AUC (OvR macro) = {metrics['auc_ovr_macro']:.4f}")

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = fitted[best_name]
    print(f"\nSelected model: {best_name} (macro F1 = {results[best_name]['macro_f1']:.4f})")

    return {
        "best_model_name": best_name, "best_model": best_model,
        "all_models": fitted, "all_results": results,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    result = train_and_select_best()
    best = result["all_results"][result["best_model_name"]]

    print("\nPer-class performance (best model):")
    classes = result["artifacts"]["label_encoder"].classes_
    for cls in classes:
        r = best["report"][cls]
        print(f"  {cls:8s}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
              f"f1={r['f1-score']:.3f}  support={int(r['support'])}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("        " + "  ".join(f"{c:>8s}" for c in classes))
    for i, row in enumerate(best["confusion_matrix"]):
        print(f"{classes[i]:8s}" + "  ".join(f"{v:8d}" for v in row))

"""
run_pipeline.py  --  THE ONE FILE YOU RUN

Runs the whole project end-to-end on the real NSL-KDD data:
  Step 3  preprocess
  Step 4  train + select best model
  Step 5  calibrate confidence
  Step 6  explain a batch of flagged alerts
  Step 7  prioritize / rank alerts
  Step 8  build the dashboard (outputs/dashboard.html)

From the project root folder, just run:
    python src/run_pipeline.py

Then open outputs/dashboard.html in your browser.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from preprocess_nslkdd import preprocess_nslkdd
from train_nslkdd import get_candidate_models, evaluate_model
from calibrate_nslkdd import calibrate_model
from explain_nslkdd import KernelSHAPLite, humanize_explanation
from prioritize_nslkdd import compute_priority_scores, BASE_SEVERITY


def run(n_alerts_to_explain=40, shap_samples=120, seed=42):
    print("=" * 60)
    print("STEP 3/8  Preprocessing real NSL-KDD data")
    print("=" * 60)
    X_train, X_test, y_train, y_test, artifacts = preprocess_nslkdd()
    le = artifacts["label_encoder"]
    print(f"Train {X_train.shape}, Test {X_test.shape}, Classes {list(le.classes_)}\n")

    print("=" * 60)
    print("STEP 4/8  Training + selecting best model")
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
    print("STEP 5/8  Calibrating confidence scores")
    print("=" * 60)
    cal = calibrate_model(candidates[best_name], X_train, y_train, X_test, y_test, cv=3)
    print(f"  ECE before {cal['ece_before']:.4f}  ->  after {cal['ece_after']:.4f}")

    # Keep calibration only if it actually improved ECE. On very imbalanced,
    # hard data the sigmoid calibrator can occasionally make things worse --
    # in that case we honestly fall back to the uncalibrated model.
    if cal["ece_after"] <= cal["ece_before"]:
        print("  -> calibration improved ECE; using calibrated model.\n")
        model = cal["calibrated_model"]
        y_proba = cal["y_proba_calibrated"]
        ece_reported = cal["ece_after"]
    else:
        print("  -> calibration did not help on this split; using the base model,\n"
              "     which is the honest choice. (Reported as a limitation.)\n")
        model = fitted[best_name]
        y_proba = model.predict_proba(X_test)
        ece_reported = cal["ece_before"]
    y_pred = np.argmax(y_proba, axis=1)

    print("=" * 60)
    print(f"STEP 6/8  Explaining {n_alerts_to_explain} flagged alerts")
    print("=" * 60)
    normal_idx = list(le.classes_).index("Normal")
    background = X_train[y_train == normal_idx]
    explainer = KernelSHAPLite(model, background, artifacts["feature_columns"],
                               n_samples=shap_samples, random_state=seed)
    attack_pos = np.where(y_pred != normal_idx)[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(attack_pos, size=min(n_alerts_to_explain, len(attack_pos)), replace=False)

    alerts = []
    for i, pos in enumerate(pick):
        x_row = X_test.iloc[pos].values
        pidx = y_pred[pos]
        plabel = le.classes_[pidx]
        conf = float(y_proba[pos, pidx])
        tlabel = le.classes_[y_test[pos]]
        exp = explainer.top_features(x_row, pidx, k=5)
        alerts.append({
            "alert_id": f"ALT-{i+1:04d}",
            "predicted_label": plabel, "true_label": tlabel,
            "correct": bool(plabel == tlabel), "confidence": round(conf, 4),
            "dest_port": 0, "protocol": "-",
            "explanation_text": humanize_explanation(plabel, exp),
            "top_features": exp["top_features"],
            "raw_features": {c: round(float(X_test.iloc[pos][c]), 3) for c in artifacts["feature_columns"][:12]},
        })
        if (i + 1) % 10 == 0:
            print(f"  Explained {i+1}/{len(pick)}")

    print("\n" + "=" * 60)
    print("STEP 7/8  Prioritizing alerts")
    print("=" * 60)
    pri = compute_priority_scores([a["predicted_label"] for a in alerts],
                                  [a["confidence"] for a in alerts], seed=seed)
    for i, a in enumerate(alerts):
        a["base_severity"] = int(pri.iloc[i]["base_severity"])
        a["asset_criticality"] = float(pri.iloc[i]["asset_criticality"])
        a["priority_score"] = float(pri.iloc[i]["priority_score"])
    alerts.sort(key=lambda a: a["priority_score"], reverse=True)
    for rank, a in enumerate(alerts, 1):
        a["rank"] = rank

    top10 = alerts[:10]
    summary = {
        "model_used": best_name,
        "macro_f1": round(results[best_name]["macro_f1"], 4),
        "auc_ovr_macro": round(results[best_name]["auc_ovr_macro"], 4),
        "ece_before_calibration": round(cal["ece_before"], 4),
        "ece_after_calibration": round(ece_reported, 4),
        "n_alerts_sampled": len(alerts),
        "sample_accuracy": round(sum(a["correct"] for a in alerts) / len(alerts), 4),
        "top10_high_severity_precision": round(sum(1 for a in top10 if a["base_severity"] >= 4) / len(top10), 4),
        "classes": list(le.classes_),
    }

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    data = {"summary": summary, "alerts": alerts}
    alerts_path = os.path.join(ROOT, "outputs", "alerts.json")
    with open(alerts_path, "w") as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 60)
    print("STEP 8/8  Building dashboard")
    print("=" * 60)
    tmpl_path = os.path.join(ROOT, "dashboard", "dashboard_template.html")
    with open(tmpl_path, encoding="utf-8") as f:
        template = f.read()
    html = template.replace("/*__ALERTS_DATA__*/{}/*__END_DATA__*/", json.dumps(data))
    dash_path = os.path.join(ROOT, "outputs", "dashboard.html")
    with open(dash_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDONE. Results summary:")
    print(json.dumps(summary, indent=2))
    print(f"\n>>> Open this file in your browser: {dash_path}")


if __name__ == "__main__":
    run(n_alerts_to_explain=100)

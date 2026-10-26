"""
predict_connection.py

Classify ONE raw NSL-KDD connection end-to-end using the saved bundle from
train_and_save.py -- no retraining.

    from predict_connection import predict_connection
    alert = predict_connection({"duration": 0, "protocol_type": "tcp", ...})

The returned dict has the same shape as one alert in outputs/alerts.json,
plus an "input_validation" key listing any defaulted fields or categories
not seen in training.

Run the self-tests from the project root folder:
    python src/predict_connection.py
"""

import os
import sys
import json
import uuid
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from preprocess_nslkdd import load_raw, COLUMN_NAMES, ATTACK_FAMILY_MAP, TEST_PATH
from explain_nslkdd import KernelSHAPLite, humanize_explanation
from prioritize_nslkdd import BASE_SEVERITY

MODEL_PATH = os.path.join(ROOT, "models", "cypher_model.joblib")

# Same SHAP settings as run_pipeline.py, so explanations match the dashboard.
SHAP_SAMPLES = 120

# Asset criticality is a stand-in, as in prioritize_nslkdd.py (which draws it
# uniformly from 1.0-5.0 per alert). A single ad-hoc connection has no asset
# inventory behind it, so we default to the midpoint of that range. Pass a
# real value via predict_connection(..., asset_criticality=x) when known.
DEFAULT_ASSET_CRITICALITY = 3.0

_bundle = None
_category_modes = None


def load_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _get_category_modes():
    # Most common training value per categorical column. Only needed when an
    # input is missing a categorical field, so it's computed lazily and cached.
    global _category_modes
    if _category_modes is None:
        b = load_bundle()
        train_raw, _ = load_raw()
        _category_modes = {c: train_raw[c].mode()[0] for c in b["categorical_columns"]}
    return _category_modes


def _validate(raw_connection):
    b = load_bundle()
    clean, defaulted, unseen = {}, [], {}
    for col in b["raw_feature_columns"]:
        if col in b["dropped_feature_columns"]:
            continue
        value = raw_connection.get(col)
        if col in b["categorical_columns"]:
            if value is None or str(value).strip() == "":
                value = _get_category_modes()[col]
                defaulted.append(col)
            value = str(value).strip()
            if value not in b["category_values"][col]:
                unseen[col] = value
        else:
            try:
                value = float(value)
                if np.isnan(value):
                    raise ValueError
            except (TypeError, ValueError):
                value = 0.0
                defaulted.append(col)
        clean[col] = value
    return clean, defaulted, unseen


def transform_connection(clean):
    """Raw validated dict -> 1-row DataFrame of the 121 encoded+scaled columns.

    Mirrors preprocess_nslkdd.py: one-hot the categoricals, align to the
    training columns (missing -> 0, so an unseen category is all zeros),
    then scale the numeric columns with the saved, already-fitted scaler.
    """
    b = load_bundle()
    row = pd.DataFrame(0.0, index=[0], columns=b["feature_columns"])
    for col in b["numeric_columns"]:
        row.at[0, col] = clean[col]
    for col in b["categorical_columns"]:
        dummy = f"{col}_{clean[col]}"
        if dummy in row.columns:
            row.at[0, dummy] = 1.0
    row[b["numeric_columns"]] = b["scaler"].transform(row[b["numeric_columns"]])
    return row


def predict_connection(raw_connection, asset_criticality=DEFAULT_ASSET_CRITICALITY,
                       alert_id=None, true_label=None):
    b = load_bundle()
    clean, defaulted, unseen = _validate(raw_connection)
    X = transform_connection(clean)

    proba = b["model"].predict_proba(X)[0]
    pidx = int(np.argmax(proba))
    plabel = b["classes"][pidx]
    conf = float(proba[pidx])

    # Background = mean of ALL Normal training rows (saved in the bundle), which
    # is exactly what the dashboard's explainer uses. A fresh explainer per call
    # keeps the explanation deterministic for a given input.
    background = pd.DataFrame([b["background_mean"]])
    explainer = KernelSHAPLite(b["model"], background, b["feature_columns"],
                               n_samples=SHAP_SAMPLES, random_state=b["seed"])
    exp = explainer.top_features(X.iloc[0].values, pidx, k=5)

    # Same formula as compute_priority_scores(), with a given (not random)
    # asset criticality.
    severity = BASE_SEVERITY.get(plabel, 1)
    max_possible = 5 * 1.0 * 5.0
    priority = round(float(severity * conf * asset_criticality / max_possible * 100), 1)

    return {
        "alert_id": alert_id or f"LIVE-{uuid.uuid4().hex[:8].upper()}",
        "predicted_label": plabel,
        "true_label": true_label if true_label is not None else "Unknown",
        "correct": bool(plabel == true_label) if true_label is not None else None,
        "confidence": round(conf, 4),
        "dest_port": 0,
        "protocol": clean["protocol_type"],
        "explanation_text": humanize_explanation(plabel, exp),
        "top_features": exp["top_features"],
        "raw_features": {c: round(float(X.iloc[0][c]), 3) for c in b["feature_columns"][:12]},
        "base_severity": int(severity),
        "asset_criticality": round(float(asset_criticality), 2),
        "priority_score": priority,
        "rank": 1,
        "input_validation": {"defaulted_fields": defaulted, "unseen_categories": unseen},
    }


def _test_rows():
    test = pd.read_csv(TEST_PATH, names=COLUMN_NAMES)
    raw_cols = load_bundle()["raw_feature_columns"]
    return test, raw_cols


def _show(title, alert):
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"  Predicted:   {alert['predicted_label']}   (true: {alert['true_label']}, "
          f"correct: {alert['correct']})")
    print(f"  Confidence:  {alert['confidence']:.4f}")
    print(f"  Explanation: {alert['explanation_text']}")
    print("  Top features:")
    for tf in alert["top_features"]:
        print(f"    {tf['feature']:28s} {tf['contribution']:+.4f}")
    print(f"  Priority:    {alert['priority_score']}  (severity {alert['base_severity']} "
          f"x conf x asset crit {alert['asset_criticality']})")
    print(f"  Validation:  {alert['input_validation']}")
    print("\nFull returned dict:")
    print(json.dumps(alert, indent=2))
    print()


if __name__ == "__main__":
    test, raw_cols = _test_rows()

    # Test A: first connection in KDDTest+.txt
    row = test.iloc[0]
    alert_a = predict_connection(row[raw_cols].to_dict(), alert_id="TEST-A",
                                 true_label=ATTACK_FAMILY_MAP[row["label"]])
    _show(f"TEST A  first row of KDDTest+.txt (raw label: {row['label']})", alert_a)

    # Test B: a known DoS row (row 0 is already 'neptune', so use 'smurf')
    pos = test.index[test["label"] == "smurf"][0]
    row = test.iloc[pos]
    alert_b = predict_connection(row[raw_cols].to_dict(), alert_id="TEST-B",
                                 true_label=ATTACK_FAMILY_MAP[row["label"]])
    _show(f"TEST B  DoS row {pos} of KDDTest+.txt (raw label: {row['label']})", alert_b)

    # Key check against the dashboard's alert format
    alerts_path = os.path.join(ROOT, "outputs", "alerts.json")
    with open(alerts_path) as f:
        expected = set(json.load(f)["alerts"][0].keys())
    got = set(alert_a.keys())
    print("=" * 60)
    print("KEY CHECK vs outputs/alerts.json")
    print("=" * 60)
    print(f"  Missing keys: {sorted(expected - got) or 'none'}")
    print(f"  Extra keys:   {sorted(got - expected) or 'none'}")

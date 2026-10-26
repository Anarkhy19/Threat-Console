"""
calibrate_nslkdd.py

Step 5: make the model's confidence scores trustworthy.

Boosted-tree and forest models are usually overconfident -- they might say
"95% sure" while being right far less often. Calibration rescales those
probabilities so the stated confidence matches real accuracy. We measure
this with Expected Calibration Error (ECE): lower is better.
"""

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV


def expected_calibration_error(y_true, y_proba, y_pred, n_bins=10):
    confidences = y_proba[np.arange(len(y_pred)), y_pred]
    correct = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def calibrate_model(base_model, X_train, y_train, X_test, y_test, method="sigmoid", cv=3):
    uncalibrated = clone(base_model)
    uncalibrated.fit(X_train, y_train)
    y_proba_raw = uncalibrated.predict_proba(X_test)
    y_pred_raw = uncalibrated.predict(X_test)
    ece_before = expected_calibration_error(y_test, y_proba_raw, y_pred_raw)

    calibrated = CalibratedClassifierCV(clone(base_model), method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    y_proba_cal = calibrated.predict_proba(X_test)
    y_pred_cal = calibrated.predict(X_test)
    ece_after = expected_calibration_error(y_test, y_proba_cal, y_pred_cal)

    return {
        "calibrated_model": calibrated,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "y_proba_calibrated": y_proba_cal,
    }

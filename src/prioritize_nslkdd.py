"""
prioritize_nslkdd.py

Step 7: rank alerts by danger, so the analyst handles the worst first.

priority_score = base_severity[attack_family] * confidence * asset_criticality

Severity values reflect how dangerous each NSL-KDD attack family is:
  - U2R (root access)      = most dangerous
  - R2L (remote break-in)  = very dangerous
  - DoS (service takedown)  = high
  - Probe (reconnaissance)  = lower (it precedes an attack, isn't the attack)
  - Normal                  = not a threat
Asset criticality is simulated here (a stand-in for a real asset inventory).
"""

import numpy as np
import pandas as pd

BASE_SEVERITY = {
    "Normal": 0,
    "Probe": 2,
    "DoS": 4,
    "R2L": 5,
    "U2R": 5,
}


def compute_priority_scores(pred_labels, confidences, seed=42):
    rng = np.random.default_rng(seed)
    severities = np.array([BASE_SEVERITY.get(lbl, 1) for lbl in pred_labels])
    # Simulated asset criticality (1.0-5.0). In production this comes from a CMDB.
    asset_crit = rng.uniform(1.0, 5.0, size=len(pred_labels))
    raw = severities * np.array(confidences) * asset_crit
    max_possible = 5 * 1.0 * 5.0
    normalized = (raw / max_possible) * 100
    return pd.DataFrame({
        "predicted_label": pred_labels,
        "confidence": confidences,
        "base_severity": severities,
        "asset_criticality": np.round(asset_crit, 2),
        "priority_score": np.round(normalized, 1),
    })

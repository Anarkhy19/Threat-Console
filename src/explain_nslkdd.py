"""
explain_nslkdd.py

Step 6: make the model explain itself.

For a single flagged alert, this computes how much each feature pushed the
model toward its decision -- a from-scratch implementation of Kernel SHAP
(Lundberg & Lee, 2017). Output: the top contributing features per alert,
turned into a plain-English sentence.

NOTE: On your laptop you can optionally replace this with the real `shap`
library for speed and rigor:
    pip install shap
    import shap
    explainer = shap.TreeExplainer(model)   # for tree models
The rest of the pipeline consumes the same {feature: contribution} format,
so nothing downstream changes. This from-scratch version is kept so the
project runs even without shap installed.
"""

import numpy as np
import pandas as pd


class KernelSHAPLite:
    def __init__(self, model, background_data, feature_names, n_samples=150, random_state=42):
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.background_mean = background_data[feature_names].mean().values
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

    def _sample_coalitions(self):
        sizes = self.rng.integers(1, self.n_features, size=self.n_samples)
        coalitions = np.zeros((self.n_samples, self.n_features), dtype=bool)
        for i, size in enumerate(sizes):
            idx = self.rng.choice(self.n_features, size=size, replace=False)
            coalitions[i, idx] = True
        return coalitions

    def explain_instance(self, x_row, target_class_idx):
        coalitions = self._sample_coalitions()
        hybrid = np.where(coalitions, x_row, self.background_mean)
        proba = self.model.predict_proba(
            pd.DataFrame(hybrid, columns=self.feature_names)
        )[:, target_class_idx]
        baseline = self.model.predict_proba(
            pd.DataFrame([self.background_mean], columns=self.feature_names)
        )[0, target_class_idx]

        A = coalitions.astype(float)
        b = proba - baseline
        lam = 1e-3
        AtA = A.T @ A + lam * np.eye(self.n_features)
        Atb = A.T @ b
        contributions = np.linalg.solve(AtA, Atb)

        full_pred = self.model.predict_proba(
            pd.DataFrame([x_row], columns=self.feature_names)
        )[0, target_class_idx]

        return {
            "contributions": dict(zip(self.feature_names, contributions)),
            "baseline_proba": float(baseline),
            "predicted_proba": float(full_pred),
        }

    def top_features(self, x_row, target_class_idx, k=5):
        result = self.explain_instance(x_row, target_class_idx)
        sorted_feats = sorted(
            result["contributions"].items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:k]
        return {
            "top_features": [
                {"feature": f, "contribution": round(float(v), 4)} for f, v in sorted_feats
            ],
            "baseline_proba": round(result["baseline_proba"], 4),
            "predicted_proba": round(result["predicted_proba"], 4),
        }


def humanize_explanation(alert_label, top_features_result):
    parts = []
    for tf in top_features_result["top_features"][:3]:
        direction = "elevated" if tf["contribution"] > 0 else "reduced"
        parts.append(f"{tf['feature']} ({direction}, impact {tf['contribution']:+.3f})")
    reasons = "; ".join(parts)
    return (
        f"Classified as {alert_label} "
        f"(confidence {top_features_result['predicted_proba']*100:.1f}%). "
        f"Primary drivers: {reasons}."
    )

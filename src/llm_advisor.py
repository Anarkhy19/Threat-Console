"""
llm_advisor.py

Feature 2: turn one prediction from predict_connection() into a plain-language
threat advisory with recommended countermeasures, using Google's Gemini API.

Honesty framing: the SHAP-style feature attributions are the evidence produced
by our own model. Gemini only INTERPRETS that evidence; its explanation and
countermeasures are advisory, not authoritative.

Setup (from the project root):
    pip install google-genai python-dotenv
    copy .env.example .env      # then put your real key in .env

    python src/llm_advisor.py   # runs the two self-tests
"""

import os
import sys
import json

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# Change this if the model ID is ever retired. A fast, inexpensive Flash model.
GEMINI_MODEL = "gemini-3.5-flash-lite"
# Tried in order if the main model is overloaded or unavailable.
GEMINI_FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash"]
REQUEST_TIMEOUT_SECONDS = 30

DISCLAIMER = ("The detection and its feature attributions come from CYPHER's model; "
              "this explanation and the countermeasures are AI-generated interpretation "
              "and are advisory, not authoritative.")

# Real environment variables take precedence over the .env file.
load_dotenv(os.path.join(ROOT, ".env"))

ADVISORY_SCHEMA = {
    "type": "object",
    "properties": {
        "threat_summary": {"type": "string"},
        "why_dangerous": {"type": "string"},
        "countermeasures": {"type": "array", "items": {"type": "string"},
                            "minItems": 3, "maxItems": 5},
    },
    "required": ["threat_summary", "why_dangerous", "countermeasures"],
}

SYSTEM_INSTRUCTION = """You are a security analyst assistant inside CYPHER, an explainable
network intrusion detection system trained on the NSL-KDD dataset. Its classes are
Normal, DoS, Probe, R2L (remote-to-local) and U2R (user-to-root).

You receive one connection the model has classified, plus the top feature attributions
from a Kernel-SHAP-style explainer. Those attributions are the ground-truth evidence:
each value is how much that feature shifted the model's probability for the predicted
class, relative to a baseline of average Normal traffic (positive = pushed toward the
predicted class, negative = pushed away). Feature names follow NSL-KDD; one-hot features
look like service_http or flag_SF (the connection's service/flag).

Your job is to INTERPRET this evidence, not to second-guess it or invent new evidence.
Ground every claim in the given features; if a feature's role is unclear, say so briefly
instead of speculating. Your countermeasures are advisory.

Respond in JSON with exactly these keys:
  threat_summary  - 2-4 sentences: what this attack type is, and why THIS connection looks
                    like it, referring to the specific top features.
  why_dangerous   - 1-3 sentences on the risk if this activity is real.
  countermeasures - 3 to 5 concrete, prioritised actions (most urgent first) that an
                    analyst or admin could take, each one sentence."""


def _base_result(**fields):
    result = {
        "status": "ok",
        "threat_summary": "",
        "why_dangerous": "",
        "countermeasures": [],
        "model_used": None,
        "disclaimer": DISCLAIMER,
    }
    result.update(fields)
    return result


def _build_prompt(p):
    lines = [
        f"Predicted class: {p['predicted_label']}",
        f"Model confidence: {p['confidence'] * 100:.1f}%",
    ]
    if "base_severity" in p:
        lines.append(f"Base severity (0-5): {p['base_severity']}")
    if "priority_score" in p:
        lines.append(f"Priority score (0-100): {p['priority_score']}")
    if p.get("protocol") and p["protocol"] != "-":
        lines.append(f"Protocol: {p['protocol']}")
    lines.append("Top contributing features (feature: attribution):")
    for tf in p.get("top_features", []):
        lines.append(f"  - {tf['feature']}: {tf['contribution']:+.4f}")
    return "\n".join(lines)


def _parse_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[cleaned.find("{"):] if "{" in cleaned else cleaned
    data = json.loads(cleaned)
    countermeasures = data.get("countermeasures", [])
    if isinstance(countermeasures, str):
        countermeasures = [countermeasures]
    return {
        "threat_summary": str(data.get("threat_summary", "")).strip(),
        "why_dangerous": str(data.get("why_dangerous", "")).strip(),
        "countermeasures": [str(c).strip() for c in countermeasures if str(c).strip()],
    }


def generate_threat_advisory(prediction_result):
    label = prediction_result.get("predicted_label")

    # Normal traffic: don't invent a threat, and don't spend an API call.
    if label == "Normal":
        return _base_result(
            status="no_threat",
            threat_summary=("No malicious activity was detected: the model classified this "
                            f"connection as Normal ({prediction_result.get('confidence', 0) * 100:.1f}% "
                            "confidence)."),
            why_dangerous="Not applicable -- no threat was identified.",
            countermeasures=[],
            note="No countermeasures needed; routine monitoring is sufficient.",
        )

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _base_result(
            status="missing_key",
            error=("GEMINI_API_KEY is not set. Add it to a .env file in the project root "
                   "(see .env.example) or set it as an environment variable."),
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_SECONDS * 1000,
                # Retry briefly on quick transient errors (rate limit / "high demand").
                # Timeouts aren't retried here; we move on to a fallback model instead.
                retry_options=types.HttpRetryOptions(
                    attempts=3, initial_delay=1.0, max_delay=4.0,
                    http_status_codes=[429, 503],
                ),
            ),
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_json_schema=ADVISORY_SCHEMA,
            temperature=0.3,
        )
        prompt = _build_prompt(prediction_result)
    except Exception as e:
        return _base_result(status="api_error", model_used=GEMINI_MODEL,
                            error=f"Could not set up the Gemini client: {type(e).__name__}: {e}")

    errors = []
    for model_id in [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt, config=config)
            text = response.text or ""
        except Exception as e:
            errors.append(f"{model_id}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if not text.strip():
            errors.append(f"{model_id}: empty response")
            continue
        try:
            return _base_result(model_used=model_id, **_parse_response(text))
        except (ValueError, AttributeError):
            # Not valid JSON -- still show what the model said rather than failing.
            return _base_result(status="parse_fallback", model_used=model_id,
                                threat_summary=text.strip())

    return _base_result(status="api_error", model_used=None,
                        error="Gemini API call failed for all models -- " + " | ".join(errors))


def _show(title, prediction, advisory):
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"  Prediction:  {prediction['predicted_label']} "
          f"({prediction['confidence'] * 100:.1f}%), true: {prediction['true_label']}")
    print(f"  Status:      {advisory['status']}   Model: {advisory['model_used']}")
    if advisory.get("error"):
        print(f"  Error:       {advisory['error']}")
    print(f"\n  Threat summary:\n    {advisory['threat_summary']}")
    print(f"\n  Why dangerous:\n    {advisory['why_dangerous']}")
    print("\n  Countermeasures:")
    if advisory["countermeasures"]:
        for i, c in enumerate(advisory["countermeasures"], 1):
            print(f"    {i}. {c}")
    else:
        print(f"    (none) {advisory.get('note', '')}")
    print(f"\n  Disclaimer:  {advisory['disclaimer']}\n")


if __name__ == "__main__":
    import pandas as pd
    from predict_connection import predict_connection, load_bundle
    from preprocess_nslkdd import COLUMN_NAMES, ATTACK_FAMILY_MAP, TEST_PATH

    test = pd.read_csv(TEST_PATH, names=COLUMN_NAMES)
    raw_cols = load_bundle()["raw_feature_columns"]

    def predict_row(pos):
        row = test.iloc[pos]
        return predict_connection(row[raw_cols].to_dict(), alert_id=f"TEST-{pos}",
                                  true_label=ATTACK_FAMILY_MAP[row["label"]])

    # Test 1: a known DoS row (first 'smurf' row in KDDTest+)
    pos = test.index[test["label"] == "smurf"][0]
    pred = predict_row(pos)
    _show(f"TEST 1  DoS row {pos} (raw label: smurf)", pred, generate_threat_advisory(pred))

    # Test 2: a Normal row (first 'normal' row in KDDTest+)
    pos = test.index[test["label"] == "normal"][0]
    pred = predict_row(pos)
    _show(f"TEST 2  Normal row {pos}", pred, generate_threat_advisory(pred))

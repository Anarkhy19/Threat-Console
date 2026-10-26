"""
server.py  --  CYPHER live prediction server

Exposes the saved model (via src/predict_connection.py) over HTTP so the
dashboard can classify single connections live.

From the project root folder, run:
    python server.py

Then open http://127.0.0.1:5000 in your browser.

Endpoints:
    GET  /health             model name + classes
    GET  /schema             field metadata for the manual-entry form
    GET  /samples?n=6        held-out KDDTest+ connections spanning families
    POST /predict            {raw_connection, true_label?, asset_criticality?}
    POST /advisory           a prediction dict (as /predict returns) -> Gemini advisory
    GET  /                   outputs/dashboard.html
    GET  /<file>             other files in outputs/ (e.g. alerts.json)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
sys.path.insert(0, os.path.join(ROOT, "src"))

import predict_connection as pc
from llm_advisor import generate_threat_advisory
from preprocess_nslkdd import COLUMN_NAMES, ATTACK_FAMILY_MAP, TEST_PATH

# Load everything once at startup so requests are fast.
BUNDLE = pc.load_bundle()
CATEGORY_MODES = pc._get_category_modes()
TEST_DF = pd.read_csv(TEST_PATH, names=COLUMN_NAMES)
TEST_DF["attack_family"] = TEST_DF["label"].map(ATTACK_FAMILY_MAP)
SAMPLE_FAMILY_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]

app = Flask(__name__)
CORS(app)


def _error(message, status):
    return jsonify({"error": message, "status": status}), status


@app.errorhandler(HTTPException)
def handle_http_error(e):
    return _error(e.description, e.code)


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception("Unhandled error")
    return _error(f"{type(e).__name__}: {e}", 500)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": BUNDLE["model_name"], "classes": BUNDLE["classes"]})


@app.get("/schema")
def schema():
    categorical = BUNDLE["categorical_columns"]
    fields = []
    for col in BUNDLE["raw_feature_columns"]:
        if col in categorical:
            fields.append({"name": col, "type": "categorical",
                           "allowed_values": BUNDLE["category_values"][col],
                           "default": CATEGORY_MODES[col]})
        else:
            fields.append({"name": col, "type": "numeric", "default": 0,
                           "used_by_model": col not in BUNDLE["dropped_feature_columns"]})
    return jsonify({
        "raw_feature_columns": BUNDLE["raw_feature_columns"],
        "categorical_columns": categorical,
        "numeric_columns": [c for c in BUNDLE["raw_feature_columns"] if c not in categorical],
        "fields": fields,
    })


@app.get("/samples")
def samples():
    try:
        n = int(request.args.get("n", 6))
        seed = int(request.args.get("seed", 42))
    except ValueError:
        return _error("'n' and 'seed' must be integers", 400)
    if not 1 <= n <= 100:
        return _error("'n' must be between 1 and 100", 400)

    # Round-robin over families so the picks span Normal/DoS/Probe/R2L/U2R.
    rng = np.random.default_rng(seed)
    pools = {}
    for fam in SAMPLE_FAMILY_ORDER:
        idx = TEST_DF.index[TEST_DF["attack_family"] == fam].to_numpy()
        pools[fam] = list(rng.permutation(idx))
    picked = []
    while len(picked) < n and any(pools.values()):
        for fam in SAMPLE_FAMILY_ORDER:
            if pools[fam] and len(picked) < n:
                picked.append(pools[fam].pop())

    raw_cols = BUNDLE["raw_feature_columns"]
    items = []
    for i in picked:
        row = TEST_DF.loc[i]
        items.append({
            "id": f"KDDTest-{i}",
            "raw_connection": json.loads(row[raw_cols].to_json()),
            "true_label": row["attack_family"],
        })
    return jsonify(items)


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error("Request body must be a JSON object", 400)
    raw = body.get("raw_connection")
    if not isinstance(raw, dict):
        return _error("'raw_connection' must be a JSON object of raw NSL-KDD features", 400)

    true_label = body.get("true_label")
    if true_label is not None and true_label not in BUNDLE["classes"]:
        return _error(f"'true_label' must be one of {BUNDLE['classes']}", 400)

    asset_crit = body.get("asset_criticality", pc.DEFAULT_ASSET_CRITICALITY)
    try:
        asset_crit = float(asset_crit)
    except (TypeError, ValueError):
        return _error("'asset_criticality' must be a number", 400)
    if not 1.0 <= asset_crit <= 5.0:
        return _error("'asset_criticality' must be between 1.0 and 5.0", 400)

    return jsonify(pc.predict_connection(raw, true_label=true_label,
                                         asset_criticality=asset_crit))


# Advisor statuses that mean "no advisory could be produced" -> HTTP error code.
ADVISORY_ERROR_CODES = {"missing_key": 503, "api_error": 502}


@app.post("/advisory")
def advisory():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error("Request body must be a JSON object (a prediction result)", 400)
    if body.get("predicted_label") not in BUNDLE["classes"]:
        return _error(f"'predicted_label' must be one of {BUNDLE['classes']}", 400)
    if not isinstance(body.get("confidence"), (int, float)):
        return _error("'confidence' must be a number", 400)
    if not isinstance(body.get("top_features", []), list):
        return _error("'top_features' must be a list of {feature, contribution}", 400)

    result = generate_threat_advisory(body)
    return jsonify(result), ADVISORY_ERROR_CODES.get(result.get("status"), 200)


@app.get("/")
def index():
    if not os.path.exists(os.path.join(OUTPUTS_DIR, "dashboard.html")):
        return _error("outputs/dashboard.html not found -- run python src/run_pipeline.py first", 404)
    return send_from_directory(OUTPUTS_DIR, "dashboard.html")


@app.get("/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"CYPHER server running -> open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)

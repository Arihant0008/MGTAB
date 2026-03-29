"""
app.py — Flask microservice for RGCN bot-detection inference.

Endpoints
---------
GET  /health            → service health check
POST /predict           → single username prediction
POST /predict/batch     → batch predictions (list of usernames)
GET  /stats             → dataset statistics
"""

import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_by_username, predict_by_index
from model_loader import get_dataset_stats, get_total_nodes

app = Flask(__name__)
CORS(app)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "mgtab-inference",
        "totalNodes": get_total_nodes(),
    })


# ── Single Prediction ────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_single():
    body = request.get_json(silent=True) or {}
    username = body.get("username", "").strip()
    node_index = body.get("nodeIndex")

    if node_index is not None:
        try:
            node_index = int(node_index)
        except (ValueError, TypeError):
            return jsonify({"error": "nodeIndex must be an integer"}), 400
        start = time.perf_counter()
        result = predict_by_index(node_index)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        if result is None:
            return jsonify({"error": f"nodeIndex {node_index} out of range"}), 400
        result["inferenceTimeMs"] = elapsed
        return jsonify(result)

    if not username:
        return jsonify({"error": "username or nodeIndex is required"}), 400

    start = time.perf_counter()
    result = predict_by_username(username)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    if result is None:
        return jsonify({"error": "Prediction failed"}), 500

    result["inferenceTimeMs"] = elapsed
    return jsonify(result)


# ── Batch Prediction ─────────────────────────────────────────────────────────
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    body = request.get_json(silent=True) or {}
    usernames = body.get("usernames", [])

    if not isinstance(usernames, list) or len(usernames) == 0:
        return jsonify({"error": "usernames must be a non-empty list"}), 400

    if len(usernames) > 500:
        return jsonify({"error": "Maximum 500 usernames per batch"}), 400

    results = []
    start = time.perf_counter()
    for uname in usernames:
        uname = str(uname).strip()
        if uname:
            pred = predict_by_username(uname)
            if pred:
                results.append(pred)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    return jsonify({
        "results": results,
        "totalProcessed": len(results),
        "totalTimeMs": elapsed,
    })


# ── Dataset Stats ─────────────────────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(get_dataset_stats())


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[inference] Starting Flask server on port 5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)

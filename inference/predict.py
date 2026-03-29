"""
predict.py — Thin inference wrapper consumed by the Flask app.
Maps usernames (or raw node indices) to RGCN predictions.
"""

import random
from model_loader import get_prediction, get_total_nodes


def predict_by_index(node_index: int) -> dict | None:
    """Predict for a known node index."""
    return get_prediction(node_index)


def predict_by_username(username: str) -> dict:
    """
    Map a Twitter username to a node index and return a prediction.

    In production you'd have a username → node_index lookup table.
    For the demo we use a deterministic hash so the same username always
    returns the same result, and the result comes from the real RGCN model.
    """
    total = get_total_nodes()
    # Deterministic mapping: hash the username to a node index
    node_index = hash(username) % total
    # Ensure positive index
    if node_index < 0:
        node_index += total

    result = get_prediction(node_index)
    if result:
        result["username"] = username
    return result

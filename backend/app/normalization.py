"""
Feature normalization constants and functions.
The min/max values are taken directly from the official MGTAB repository:
https://github.com/GraphDetec/MGTAB/tree/main/Standardization
Normalization pipeline (matching the paper):
 1. Numerical counts → log(1 + x)           [except lengths & created_at]
 2. MinMaxScaler using the bounds below
 3. Booleans → 0.0 / 1.0  (no scaling)
"""

import math
import numpy as np

# ── MinMax Bounds (from labeled data) ─────────────────────────────────
# These values correspond to the ALREADY log-transformed features.
# e.g. followers_count max 25.57 = log(1 + some_huge_number)

FEATURE_BOUNDS = {
    "followers_count":          {"min": 0.0,       "max": 25.572674},
    "friends_count":            {"min": 0.0,       "max": 21.029877},
    "listed_count":             {"min": 0.0,       "max": 17.675406},
    "created_at":               {"min": 36.553529, "max": 51.711108},
    "favourites_count":         {"min": 0.0,       "max": 19.711042},
    "statuses_count":           {"min": 0.0,       "max": 20.386231},
    "screen_name_length":       {"min": 3.0,       "max": 15.0},
    "name_length":              {"min": 1.0,       "max": 50.0},
    "description_length":       {"min": 0.0,       "max": 204.0},
    "followers_friends_ratios": {"min": 0.0,       "max": 11.169299},
}

# Features that need log(1+x) transformation before MinMax scaling
LOG_TRANSFORM_FEATURES = {
    "followers_count",
    "friends_count",
    "listed_count",
    "favourites_count",
    "statuses_count",
    "followers_friends_ratios",
}

# created_at also gets log-transformed, but its formula is different
# The dataset uses log(timestamp_seconds) which gives values in ~36-51 range


def log_transform(value: float) -> float:
    """Apply log(1 + x) transformation."""
    return math.log1p(max(0.0, value))


def minmax_scale(value: float, feat_min: float, feat_max: float) -> float:
    """Scale value to [0, 1] range using min-max normalization."""
    if feat_max <= feat_min:
        return 0.0
    scaled = (value - feat_min) / (feat_max - feat_min)
    return float(np.clip(scaled, 0.0, 1.0))


def normalize_numerical_feature(name: str, raw_value: float) -> float:
    """
    Full normalization pipeline for a single numerical feature.
    1. Log-transform (if applicable)
    2. MinMax scale
    """
    bounds = FEATURE_BOUNDS[name]

    if name in LOG_TRANSFORM_FEATURES:
        value = log_transform(raw_value)
    elif name == "created_at":
        # created_at is already expected as log(timestamp) from the caller
        value = raw_value
    else:
        # screen_name_length, name_length, description_length
        value = float(raw_value)

    return minmax_scale(value, bounds["min"], bounds["max"])


def encode_created_at(timestamp_seconds: float) -> float:
    """
    Convert a Unix timestamp (seconds since epoch) to the log-scaled
    value used in the MGTAB dataset.
    
    The dataset stores created_at with MinMax bounds [36.55, 51.71].
    Testing reveals that log(unix_seconds) ≈ 21 falls BELOW the minimum,
    while log(unix_seconds * 1e9) ≈ 41.8 falls correctly within bounds.
    
    This means the MGTAB dataset encoded created_at as:
        log(timestamp_nanoseconds) = log(timestamp_seconds × 1e9)
    
    Examples:
        Jan 2008: log(1.20e18) ≈ 41.63 → scaled ≈ 0.33
        Mar 2015: log(1.43e18) ≈ 41.80 → scaled ≈ 0.35
        Nov 2023: log(1.70e18) ≈ 41.97 → scaled ≈ 0.36
    """
    if timestamp_seconds <= 0:
        return FEATURE_BOUNDS["created_at"]["min"]
    # MGTAB uses log(timestamp in nanoseconds)
    return math.log(timestamp_seconds * 1e9)
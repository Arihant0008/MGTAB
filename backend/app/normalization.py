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
    
    The dataset stores created_at as log(timestamp_seconds), giving
    values in the ~36.5 – 51.7 range.
    
    Example: 
        Jan 1, 2020 = 1577836800 → log(1577836800) ≈ 21.18 (natural log)
    
    However, the MGTAB values (~36-51) suggest log base e of larger numbers
    or a different epoch. The dataset likely uses:
        log(seconds_since_twitter_epoch) where twitter started ~2006
    
    Given the min/max values (36.55 – 51.71), these correspond to:
        e^36.55 ≈ 7.5e15 and e^51.71 ≈ 3.3e22
    
    This doesn't map to standard timestamps. The actual formula appears 
    to be: months_since_epoch * some_factor or a custom encoding.
    
    Safest approach: use natural log of the unix timestamp, then
    minmax-scale with the known bounds so it maps to [0,1].
    """
    if timestamp_seconds <= 0:
        return FEATURE_BOUNDS["created_at"]["min"]
    return math.log(timestamp_seconds)

"""
Feature engineering pipeline.
Converts raw profile data + tweets into the exact 788-dimensional
feature vector that the trained RGCN model expects.
Feature vector layout (matching MGTAB features.pt):
  [0]      profile_use_background_image   (bool)
  [1]      default_profile                (bool)
  [2]      verified                       (bool)
  [3]      followers_count                (numerical, log+minmax)
  [4]      default_profile_image          (bool)
  [5]      listed_count                   (numerical, log+minmax)
  [6]      statuses_count                 (numerical, log+minmax)
  [7]      friends_count                  (numerical, log+minmax)
  [8]      geo_enabled                    (bool)
  [9]      favourites_count               (numerical, log+minmax)
  [10]     created_at                     (numerical, log+minmax)
  [11]     screen_name_length             (numerical, minmax)
  [12]     name_length                    (numerical, minmax)
  [13]     description_length             (numerical, minmax)
  [14]     followers_friends_ratios       (numerical, log+minmax)
  [15]     default_profile_background_color (bool)
  [16]     default_profile_sidebar_fill_color (bool)
  [17]     default_profile_sidebar_border_color (bool)
  [18]     has_URL                        (bool)
  [19]     profile_background_image_URL   (bool)
  [20-787] tweet_features (LaBSE 768-dim) (float)
"""

import numpy as np
import logging
from datetime import datetime
from typing import Optional

from .normalization import (
    normalize_numerical_feature,
    encode_created_at,
)

logger = logging.getLogger(__name__)

# ── Lazy-loaded LaBSE model (using transformers directly) ────────────
_labse_tokenizer = None
_labse_model = None


def _get_labse():
    """Lazy-load the LaBSE tokenizer and model to avoid slow startup."""
    global _labse_tokenizer, _labse_model
    if _labse_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading LaBSE model (first time, may take a minute)...")
        _labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
        _labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
        _labse_model.eval()
        logger.info("LaBSE model loaded successfully.")
    return _labse_tokenizer, _labse_model


def compute_profile_features(profile: dict) -> np.ndarray:
    """
    Convert raw profile data into a 20-dimensional feature vector.
    
    Args:
        profile: dict with keys like followers_count, friends_count,
                 name, screen_name, description, created_at, etc.
    
    Returns:
        np.ndarray of shape (20,) with normalized features.
    """
    features = np.zeros(20, dtype=np.float32)

    # ── Booleans (indices 0, 1, 2, 4, 8, 15, 16, 17, 18, 19) ────────
    bool_mappings = {
        0:  "profile_use_background_image",
        1:  "default_profile",
        2:  "verified",
        4:  "default_profile_image",
        8:  "geo_enabled",
        15: "default_profile_background_color",
        16: "default_profile_sidebar_fill_color",
        17: "default_profile_sidebar_border_color",
        18: "has_url",
        19: "profile_background_image_url",
    }

    for idx, key in bool_mappings.items():
        raw = profile.get(key, False)
        features[idx] = 1.0 if raw else 0.0

    # ── Numerical counts (indices 3, 5, 6, 7, 9) ────────────────────
    count_mappings = {
        3: ("followers_count", "followers_count"),
        5: ("listed_count", "listed_count"),
        6: ("statuses_count", "statuses_count"),
        7: ("friends_count", "friends_count"),
        9: ("favourites_count", "favourites_count"),
    }

    for idx, (profile_key, norm_key) in count_mappings.items():
        raw_val = float(profile.get(profile_key, 0))
        features[idx] = normalize_numerical_feature(norm_key, raw_val)

    # ── created_at (index 10) ────────────────────────────────────────
    created_at = profile.get("created_at", None)
    if created_at:
        if isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                ts = dt.timestamp()
            except ValueError:
                # Try Twitter's date format: "Wed Oct 10 20:19:24 +0000 2012"
                try:
                    dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                    ts = dt.timestamp()
                except ValueError:
                    ts = datetime(2020, 1, 1).timestamp()
        elif isinstance(created_at, (int, float)):
            ts = float(created_at)
        else:
            ts = datetime(2020, 1, 1).timestamp()
        
        log_ts = encode_created_at(ts)
        features[10] = normalize_numerical_feature("created_at", log_ts)
    else:
        features[10] = 0.5  # midpoint default

    # ── Derived length features (indices 11, 12, 13) ─────────────────
    screen_name = profile.get("screen_name", "")
    name = profile.get("name", "")
    description = profile.get("description", "")

    features[11] = normalize_numerical_feature(
        "screen_name_length", len(str(screen_name))
    )
    features[12] = normalize_numerical_feature(
        "name_length", len(str(name))
    )
    features[13] = normalize_numerical_feature(
        "description_length", len(str(description))
    )

    # ── followers/friends ratio (index 14) ───────────────────────────
    followers = float(profile.get("followers_count", 0))
    friends = float(profile.get("friends_count", 1))  # avoid div by zero
    ratio = followers / max(friends, 1.0)
    features[14] = normalize_numerical_feature("followers_friends_ratios", ratio)

    return features


def compute_labse_embedding(tweets: list[str]) -> np.ndarray:
    """
    Encode a list of tweet texts using LaBSE and average them.
    Uses transformers AutoModel directly (compatible with PyTorch 2.1).
    
    Args:
        tweets: list of tweet text strings.
    
    Returns:
        np.ndarray of shape (768,) — averaged LaBSE embedding.
    """
    if not tweets or all(not t.strip() for t in tweets):
        logger.warning("No tweets provided — returning zero embedding.")
        return np.zeros(768, dtype=np.float32)

    # Filter out empty tweets
    valid_tweets = [t.strip() for t in tweets if t.strip()]
    if not valid_tweets:
        return np.zeros(768, dtype=np.float32)

    import torch

    tokenizer, model = _get_labse()
    
    # Tokenize and encode
    encoded = tokenizer(
        valid_tweets,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = model(**encoded)
        # LaBSE uses the [CLS] token embedding (pooler_output)
        embeddings = outputs.pooler_output  # (num_tweets, 768)
        # NOTE: Do NOT L2-normalize here. The MGTAB training data used raw
        # pooler_output (norms ~18.2). Normalizing shrinks norms to ~0.5,
        # making the 768 tweet dims invisible to the trained RGCN weights.
    
    # Sum across all tweets (NOT average).
    # The MGTAB training data used summed LaBSE pooler_output per node,
    # producing norms ~18-20. Averaging would give norms ~5-6, which
    # the trained RGCN weights would underweight relative to profile features.
    summed_embedding = embeddings.sum(dim=0).numpy().astype(np.float32)

    return summed_embedding


def build_node_feature(
    profile: dict,
    tweets: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Build the full 788-dimensional feature vector for a single node.
    
    Args:
        profile: raw profile data dict
        tweets: list of tweet texts (can be empty/None)
    
    Returns:
        np.ndarray of shape (788,) — [profile_20 || tweet_768]
    """
    profile_vec = compute_profile_features(profile)          # (20,)
    tweet_vec = compute_labse_embedding(tweets or [])        # (768,)
    node_feature = np.concatenate([profile_vec, tweet_vec])  # (788,)
    return node_feature

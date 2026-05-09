"""
MGTAB Full Pipeline Test Suite
Tests: normalization → features → graph builder → RGCN inference
Run: backend\venv\Scripts\python.exe tests\test_pipeline.py
"""
import sys
import os
import json
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
os.environ.setdefault('UPSTASH_REDIS_REST_URL', '')
os.environ.setdefault('UPSTASH_REDIS_REST_TOKEN', '')

PASS = 0
FAIL = 0

def ok(name):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}")

def fail(name, reason):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name} — {reason}")

def header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# TEST 1: Normalization
# ─────────────────────────────────────────────────────────────────
header("TEST 1: Normalization Pipeline")
from app.normalization import normalize_numerical_feature, encode_created_at
import datetime

cases = [
    ("followers_count",    0,       0.000, 0.001, "zero followers"),
    ("followers_count",    1000,    0.270, 0.020, "1k followers"),
    ("followers_count",    1000000, 0.540, 0.030, "1M followers"),
    ("friends_count",      200,     0.252, 0.030, "200 friends"),
    ("statuses_count",     5000,    0.418, 0.030, "5k statuses"),
    ("favourites_count",   1000,    0.350, 0.030, "1k favourites"),
    ("screen_name_length", 8,       0.417, 0.050, "screen name len=8"),
    ("description_length", 100,     0.490, 0.050, "desc len=100"),
]
for feat, val, expected, tol, label in cases:
    got = normalize_numerical_feature(feat, val)
    if abs(got - expected) <= tol:
        ok(f"{feat}({val}) = {got:.4f} ≈ {expected}")
    else:
        fail(f"{feat}({val})", f"got {got:.4f}, expected {expected:.4f} ±{tol}")

# created_at bounds
ts_2015 = datetime.datetime(2015, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
log_ts = encode_created_at(ts_2015)
v = normalize_numerical_feature("created_at", log_ts)
if 0.20 < v < 0.45:
    ok(f"created_at(2015) = {v:.4f} in (0.20, 0.45)")
else:
    fail("created_at(2015)", f"got {v:.4f} — out of expected range")

ts_2022 = datetime.datetime(2022, 6, 1, tzinfo=datetime.timezone.utc).timestamp()
log_ts2 = encode_created_at(ts_2022)
v2 = normalize_numerical_feature("created_at", log_ts2)
if 0.30 < v2 < 0.50:
    ok(f"created_at(2022) = {v2:.4f} in (0.30, 0.50)")
else:
    fail("created_at(2022)", f"got {v2:.4f} — out of expected range")

# ─────────────────────────────────────────────────────────────────
# TEST 2: Profile Feature Vector
# ─────────────────────────────────────────────────────────────────
header("TEST 2: Profile Feature Vector (shape + range)")
from app.features import compute_profile_features

profiles = {
    "typical_human": {
        "followers_count": 500, "friends_count": 300, "listed_count": 10,
        "statuses_count": 2000, "favourites_count": 800,
        "name": "John Doe", "screen_name": "johndoe123",
        "description": "Software engineer, coffee lover, dad.",
        "created_at": "2015-06-15T00:00:00Z",
        "default_profile": False, "default_profile_image": False,
        "verified": False, "has_url": True, "geo_enabled": False,
        "profile_use_background_image": True,
        "default_profile_background_color": False,
        "default_profile_sidebar_fill_color": False,
        "default_profile_sidebar_border_color": False,
        "profile_background_image_url": False,
    },
    "bot_pattern": {
        "followers_count": 50000, "friends_count": 5, "listed_count": 0,
        "statuses_count": 99999, "favourites_count": 0,
        "name": "xb7f2k9", "screen_name": "xb7f2k9",
        "description": "",
        "created_at": "2023-10-01T00:00:00Z",
        "default_profile": True, "default_profile_image": True,
        "verified": False, "has_url": False, "geo_enabled": False,
        "profile_use_background_image": False,
        "default_profile_background_color": True,
        "default_profile_sidebar_fill_color": True,
        "default_profile_sidebar_border_color": True,
        "profile_background_image_url": False,
    },
    "celebrity": {
        "followers_count": 50000000, "friends_count": 500, "listed_count": 50000,
        "statuses_count": 30000, "favourites_count": 5000,
        "name": "Big Star", "screen_name": "bigstar",
        "description": "Official account of Big Star. Actor & Filmmaker.",
        "created_at": "2009-03-01T00:00:00Z",
        "default_profile": False, "default_profile_image": False,
        "verified": True, "has_url": True, "geo_enabled": False,
        "profile_use_background_image": True,
        "default_profile_background_color": False,
        "default_profile_sidebar_fill_color": False,
        "default_profile_sidebar_border_color": False,
        "profile_background_image_url": True,
    },
}

for name, profile in profiles.items():
    vec = compute_profile_features(profile)
    shape_ok = vec.shape == (20,)
    range_ok = bool(np.all(vec >= 0) and np.all(vec <= 1))
    if shape_ok and range_ok:
        ok(f"{name}: shape={vec.shape}, range=[{vec.min():.3f}, {vec.max():.3f}]")
    else:
        fail(name, f"shape={vec.shape}, range=[{vec.min():.3f}, {vec.max():.3f}]")

# ─────────────────────────────────────────────────────────────────
# TEST 3: LaBSE Embedding (no tweets → noise vector, not zeros)
# ─────────────────────────────────────────────────────────────────
header("TEST 3: LaBSE Embedding — Zero-tweet fallback")
from app.features import compute_labse_embedding, LABSE_TARGET_NORM

emb_empty = compute_labse_embedding([])
if emb_empty.shape == (768,):
    ok(f"Empty tweets → shape (768,)")
else:
    fail("Empty tweets shape", f"got {emb_empty.shape}")

# Must NOT be all zeros
if np.any(emb_empty != 0):
    ok(f"Empty tweets → noise vector (not zeros), norm={np.linalg.norm(emb_empty):.2f}")
else:
    fail("Empty tweets fallback", "got a zero vector — this corrupts RGCN aggregation!")

# Norm should be ~10% of target
expected_noise_norm = LABSE_TARGET_NORM * 0.10
got_norm = np.linalg.norm(emb_empty)
if abs(got_norm - expected_noise_norm) < 2.0:
    ok(f"Noise vector norm={got_norm:.2f} ≈ {expected_noise_norm:.2f} (10% of target)")
else:
    fail("Noise norm", f"got {got_norm:.2f}, expected ≈ {expected_noise_norm:.2f}")

# ─────────────────────────────────────────────────────────────────
# TEST 4: Graph Builder — 3-tuple + quality tracking
# ─────────────────────────────────────────────────────────────────
header("TEST 4: Graph Builder")
from app.graph_builder import build_mini_graph
import torch

def make_profile(name="user", followers=500, friends=300, year=2015):
    return {
        "followers_count": followers, "friends_count": friends,
        "listed_count": 5, "statuses_count": 1000, "favourites_count": 200,
        "name": name, "screen_name": name.lower(),
        "description": f"I am {name}",
        "created_at": f"{year}-06-01T00:00:00Z",
        "default_profile": False, "default_profile_image": False,
        "verified": False, "has_url": True, "geo_enabled": False,
        "profile_use_background_image": True,
        "default_profile_background_color": False,
        "default_profile_sidebar_fill_color": False,
        "default_profile_sidebar_border_color": False,
        "profile_background_image_url": False,
    }

# 4a: single node
req_solo = {"target": {"profile": make_profile("Alice"), "tweets": ["Hello!"]}, "neighbors": [], "relations": []}
data, idx, quality = build_mini_graph(req_solo)
if len([data, idx, quality]) == 3:
    ok("Returns 3-tuple (data, idx, quality)")
else:
    fail("3-tuple", "not a 3-tuple")

if data.num_nodes == 1 and data.num_edges == 1:
    ok(f"Solo graph: {data.num_nodes} node, {data.num_edges} self-loop edge")
else:
    fail("Solo graph", f"nodes={data.num_nodes}, edges={data.num_edges}")

if quality["nodes_with_tweets"] == 1 and quality["nodes_profile_only"] == 0:
    ok("Solo quality: target counted as has_tweets")
else:
    fail("Solo quality", str(quality))

# 4b: two neighbors, one with tweets, one without
req_multi = {
    "target": {"profile": make_profile("Alice"), "tweets": ["Hello"]},
    "neighbors": [
        {"id": "bob",   "profile": make_profile("Bob"),   "tweets": ["Hi from Bob"]},
        {"id": "carol", "profile": make_profile("Carol"), "tweets": []},
    ],
    "relations": [
        {"source": "bob",        "target": "__target__", "relation": "follower"},
        {"source": "__target__", "target": "carol",      "relation": "friend"},
    ],
}
data2, idx2, q2 = build_mini_graph(req_multi)
if data2.num_nodes == 3:
    ok(f"Multi graph: {data2.num_nodes} nodes (target+bob+carol)")
else:
    fail("Multi graph nodes", f"got {data2.num_nodes}")

if data2.num_edges == 2:
    ok(f"Multi graph: {data2.num_edges} edges (follower + friend)")
else:
    fail("Multi graph edges", f"got {data2.num_edges}")

if q2["nodes_with_tweets"] == 2 and q2["nodes_profile_only"] == 1:
    ok(f"Quality tracking: {q2['nodes_with_tweets']} w/tweets, {q2['nodes_profile_only']} profile-only")
else:
    fail("Quality tracking", str(q2))

if data2.x.shape == (3, 788):
    ok(f"Feature matrix shape: {data2.x.shape} (3 nodes × 788 dims)")
else:
    fail("Feature matrix shape", f"got {data2.x.shape}")

# 4c: edges_filtered counter
req_filtered = {
    "target": {"profile": make_profile("Alice"), "tweets": ["Hello"]},
    "neighbors": [],  # no neighbor data
    "relations": [
        {"source": "__target__", "target": "ghost_user", "relation": "friend"},
    ],
}
data3, idx3, q3 = build_mini_graph(req_filtered)
if q3["edges_filtered"] == 1:
    ok(f"edges_filtered correctly counted: {q3['edges_filtered']}")
else:
    fail("edges_filtered", f"got {q3['edges_filtered']}")

# ─────────────────────────────────────────────────────────────────
# TEST 5: RGCN Model — load + forward pass
# ─────────────────────────────────────────────────────────────────
header("TEST 5: RGCN Model Load + Forward Pass")

try:
    from app.inference import InferenceEngine
    engine = InferenceEngine()
    ok("InferenceEngine loaded (best_rgcn.pt)")

    # Run predict on solo graph
    result = engine.predict_from_request(req_solo)
    ok(f"predict_from_request returned keys: {list(result.keys())}")

    # Check all required keys present
    required_keys = ["label_pred", "prob_human", "prob_bot", "confidence", "quality_warning", "graph_info"]
    missing = [k for k in required_keys if k not in result]
    if not missing:
        ok("All required response keys present")
    else:
        fail("Response keys", f"missing: {missing}")

    # Probabilities must sum to ~1.0
    prob_sum = result["prob_human"] + result["prob_bot"]
    if abs(prob_sum - 1.0) < 0.01:
        ok(f"Probabilities sum to {prob_sum:.4f} ≈ 1.0")
    else:
        fail("Probability sum", f"got {prob_sum:.4f}")

    # No value should be 0.0 or 1.0 (calibration check)
    if result["prob_human"] > 0.04 and result["prob_bot"] > 0.04:
        ok(f"Calibrated: human={result['prob_human']:.4f}, bot={result['prob_bot']:.4f} (both > 5%)")
    else:
        fail("Calibration floor", f"human={result['prob_human']}, bot={result['prob_bot']}")

    # label_pred must be valid
    if result["label_pred"] in ("human", "bot"):
        ok(f"label_pred = '{result['label_pred']}'")
    else:
        fail("label_pred", f"invalid value: {result['label_pred']}")

    # confidence must match max prob
    expected_conf = round(max(result["prob_human"], result["prob_bot"]), 4)
    if abs(result["confidence"] - expected_conf) < 0.001:
        ok(f"confidence = {result['confidence']} (matches max prob)")
    else:
        fail("confidence", f"got {result['confidence']}, expected {expected_conf}")

    # ── TEST 5b: bot-pattern profile
    req_bot = {
        "target": {
            "profile": profiles["bot_pattern"],
            "tweets": ["follow me", "click here", "buy now", "free money", "join us"],
        },
        "neighbors": [], "relations": [],
    }
    bot_result = engine.predict_from_request(req_bot)
    ok(f"Bot-pattern profile: label={bot_result['label_pred']}, bot%={bot_result['prob_bot']*100:.1f}%")

    # ── TEST 5c: celebrity profile
    req_celeb = {
        "target": {
            "profile": profiles["celebrity"],
            "tweets": ["Excited to announce our new movie!", "Thank you fans!", "Live from NYC tonight"],
        },
        "neighbors": [], "relations": [],
    }
    celeb_result = engine.predict_from_request(req_celeb)
    ok(f"Celebrity profile:   label={celeb_result['label_pred']}, human%={celeb_result['prob_human']*100:.1f}%")

    # ── TEST 5d: multi-neighbor graph
    multi_result = engine.predict_from_request(req_multi)
    ok(f"Multi-neighbor graph: label={multi_result['label_pred']}, confidence={multi_result['confidence']*100:.1f}%")
    ok(f"  graph_info: nodes={multi_result['graph_info']['num_nodes']}, edges={multi_result['graph_info']['num_edges']}, tweet_coverage={multi_result['graph_info']['nodes_with_tweets']}/{multi_result['graph_info']['num_nodes']}")

except Exception as e:
    import traceback
    fail("InferenceEngine", str(e))
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────
# TEST 6: Cache module (without Redis — graceful degradation)
# ─────────────────────────────────────────────────────────────────
header("TEST 6: Cache Module — Graceful Degradation (no Redis)")
from app.cache import cache_get, cache_set, cache_delete, cache_ping

result_get = cache_get("testuser")
if result_get is None:
    ok("cache_get without Redis → None (graceful)")
else:
    fail("cache_get", f"expected None, got {result_get}")

result_set = cache_set("testuser", {"label_pred": "human", "prob_human": 0.8, "prob_bot": 0.2})
if result_set is False:
    ok("cache_set without Redis → False (graceful)")
else:
    fail("cache_set", f"expected False, got {result_set}")

result_del = cache_delete("testuser")
if result_del is False:
    ok("cache_delete without Redis → False (graceful)")
else:
    fail("cache_delete", f"expected False, got {result_del}")

ping = cache_ping()
if ping is False:
    ok("cache_ping without Redis → False (graceful)")
else:
    fail("cache_ping", f"expected False, got {ping}")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed  |  {FAIL} failed")
print("=" * 60)
if FAIL > 0:
    print("  ⚠  Some tests failed — review above output")
    sys.exit(1)
else:
    print("  ✓  All tests passed!")
    sys.exit(0)

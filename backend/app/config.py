"""
Configuration constants for the MGTAB Bot Detector backend.
All paths are relative to the backend/ directory.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BASE_DIR.parent                             # MGTAB/

# In local dev, the model is in the parent "Datasets and pre..." folder.
# In production (Hugging Face), it will be copied directly into the backend/ root.
local_dev_model = PROJECT_ROOT / "Datasets and precrosessing" / "best_rgcn.pt"

if local_dev_model.exists():
    MODEL_PATH = local_dev_model
else:
    MODEL_PATH = BASE_DIR / "best_rgcn.pt"

# ── Model Architecture ────────────────────────────────────────────────
NUM_FEATURES = 788          # 20 profile + 768 tweet (LaBSE)
NUM_PROFILE_FEATURES = 20
NUM_TWEET_FEATURES = 768
NUM_CLASSES = 2             # human=0, bot=1
NUM_RELATIONS = 7           # follower, friend, mention, reply, quoted, url, hashtag
HIDDEN_DIM = 256

# ── LaBSE ─────────────────────────────────────────────────────────────
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"

# ── Relation Mapping ──────────────────────────────────────────────────
RELATION_MAP = {
    "follower": 0,
    "friend":   1,
    "mention":  2,
    "reply":    3,
    "quoted":   4,
    "url":      5,
    "hashtag":  6,
}

RELATION_NAMES = {v: k for k, v in RELATION_MAP.items()}

# ── Server ────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

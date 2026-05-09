"""
Configuration constants for the MGTAB Bot Detector backend.
All paths are relative to the backend/ directory.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load .env before reading any env vars

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
# Paper Table 4: 7 relation types with specific direction semantics
RELATION_MAP = {
    "follower": 0,   # user A is followed by user B → edge: B → A
    "friend":   1,   # user A follows user B        → edge: A → B
    "mention":  2,   # user A mentions user B       → edge: A → B
    "reply":    3,   # user A replies to user B     → edge: A → B
    "quoted":   4,   # user A quotes user B         → edge: A → B
    "url":      5,   # undirected co-occurrence     → edges: A↔B
    "hashtag":  6,   # undirected co-occurrence     → edges: A↔B
}

RELATION_NAMES = {v: k for k, v in RELATION_MAP.items()}

# Relations where the edge direction is REVERSED from how the frontend sends it.
# Paper: "follower" means "user A is followed BY user B", so B→A.
# Frontend sends source=target, target=neighbor, so we must reverse to neighbor→target.
REVERSE_SOURCE_RELATIONS = {"follower"}

# Relations that are undirected (need edges in both directions)
# Paper Table 4: URL and hashtag co-occurrence are undirected.
UNDIRECTED_RELATIONS = {"url", "hashtag"}

# ── Scweet Scraper ────────────────────────────────────────────────
# Cookie-based auth — no passwords needed.
# Get auth_token from: x.com → F12 → Application → Cookies → auth_token
TWITTER_AUTH_TOKEN = os.getenv("TWITTER_AUTH_TOKEN", "")

# Optional proxy for Scweet requests (e.g. "http://user:pass@host:port")
PROXY_URL = os.getenv("PROXY_URL", "")

MAX_NEIGHBORS_PER_RELATION = 10  # 10 × 5 relation types = ~50 neighbors max
SCRAPE_DELAY_SECONDS = float(os.getenv("SCRAPE_DELAY_SECONDS", "3.0"))
MAX_TWEETS_TARGET = 20           # tweets to fetch for the target user
MAX_TWEETS_NEIGHBOR = 5          # tweets to fetch for each neighbor

# ── Server ────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
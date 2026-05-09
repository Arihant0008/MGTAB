"""
Redis cache helper using Upstash REST API.
Uses upstash-redis (HTTPS REST) — no persistent TCP connection needed.
This makes it safe for HuggingFace Spaces which can restart containers
at any time (a TCP Redis client would lose its connection on restart).

Cache strategy:
  - Key:   prediction:{lowercase_handle}
  - Value: JSON-serialised PredictResponse dict
  - TTL:   REDIS_CACHE_TTL seconds (default 3600 = 1 hour)

Graceful degradation:
  If Redis is not configured or any error occurs, all cache operations
  silently return None / False so the app keeps working normally.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
CACHE_KEY_PREFIX = "prediction"
CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # 1 hour default

# ── Lazy-loaded Redis client ───────────────────────────────────────────
_redis_client = None


def _get_client():
    """
    Lazy-initialise the Upstash Redis client.
    Returns None if credentials are not configured — app continues normally.
    """
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip().strip('"')
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip().strip('"')

    if not url or not token:
        logger.warning(
            "Redis: UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN not set. "
            "Caching disabled — app will work normally without it."
        )
        return None

    try:
        from upstash_redis import Redis
        _redis_client = Redis(url=url, token=token)
        logger.info(f"Redis: Connected to Upstash at {url}")
        return _redis_client
    except ImportError:
        logger.warning(
            "Redis: upstash-redis package not installed. "
            "Run: pip install upstash-redis"
        )
        return None
    except Exception as e:
        logger.warning(f"Redis: Failed to initialise client: {e}")
        return None


def _make_key(handle: str) -> str:
    """Build the cache key for a Twitter handle."""
    return f"{CACHE_KEY_PREFIX}:{handle.lower().strip().lstrip('@')}"


# ── Public API ────────────────────────────────────────────────────────

def cache_get(handle: str) -> dict | None:
    """
    Retrieve a cached prediction result for a Twitter handle.
    Returns dict with prediction data if cache hit, None otherwise.
    """
    client = _get_client()
    if client is None:
        return None

    key = _make_key(handle)
    try:
        raw = client.get(key)
        if raw is None:
            logger.info(f"Redis: Cache MISS for @{handle}")
            return None
        data = json.loads(raw) if isinstance(raw, str) else raw
        logger.info(f"Redis: Cache HIT for @{handle} (key={key})")
        return data
    except Exception as e:
        logger.warning(f"Redis: cache_get failed for @{handle}: {e}")
        return None


def cache_set(handle: str, result: dict) -> bool:
    """
    Store a prediction result in the cache.
    Returns True if stored successfully, False otherwise.
    """
    client = _get_client()
    if client is None:
        return False

    key = _make_key(handle)
    try:
        import datetime
        result_to_store = {
            **result,
            "cached_at": datetime.datetime.utcnow().isoformat() + "Z",
            "from_cache": True,
        }
        client.set(key, json.dumps(result_to_store), ex=CACHE_TTL)
        logger.info(f"Redis: Cached result for @{handle} (ttl={CACHE_TTL}s)")
        return True
    except Exception as e:
        logger.warning(f"Redis: cache_set failed for @{handle}: {e}")
        return False


def cache_delete(handle: str) -> bool:
    """Delete a cached result. Returns True if deleted, False otherwise."""
    client = _get_client()
    if client is None:
        return False

    key = _make_key(handle)
    try:
        client.delete(key)
        logger.info(f"Redis: Deleted cache for @{handle} (key={key})")
        return True
    except Exception as e:
        logger.warning(f"Redis: cache_delete failed for @{handle}: {e}")
        return False


def cache_ping() -> bool:
    """Health check — returns True if Redis is reachable."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False

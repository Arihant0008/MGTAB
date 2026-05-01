"""
Scweet-based Twitter/X scraper for the MGTAB Bot Detector.
Provides automated scraping of:
  - Target user profile metadata + recent tweets
  - Ego-graph expansion: followers, friends, mentions, replies, quotes
  - Neighbor enrichment: real profile + tweet data for each neighbor
  - Opportunistic URL/hashtag co-occurrence edge discovery
Uses Scweet v5 with cookie-based auth_token authentication (no passwords).
All async methods use Scweet's aget_* variants for FastAPI compatibility.
Rate-Limit Resilience:
  All tweet-fetching calls are wrapped in explicit RateLimitError / HTTP 429
  handlers. If a neighbor's tweets are rate-limited, the node still enters the
  graph with profile features; only the content pillar (tweets) is starved.
"""

import asyncio
import logging
import re
from typing import Any, Callable, Coroutine, Optional

from .config import (
    MAX_NEIGHBORS_PER_RELATION,
    MAX_TWEETS_NEIGHBOR,
    MAX_TWEETS_TARGET,
    PROXY_URL,
    SCRAPE_DELAY_SECONDS,
    TWITTER_AUTH_TOKEN,
)

logger = logging.getLogger(__name__)


# ── Custom Exceptions ─────────────────────────────────────────────
class ScraperError(Exception):
    """Base exception for scraper errors."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class ScraperAuthError(ScraperError):
    """Twitter authentication failed."""

    def __init__(self, message: str = "Twitter authentication failed"):
        super().__init__(message, status_code=401)


class ScraperRateLimitError(ScraperError):
    """Rate limited by Twitter/X."""

    def __init__(self, message: str = "Rate limited by Twitter/X. Please try again later."):
        super().__init__(message, status_code=429)


class ScraperUserNotFoundError(ScraperError):
    """Target user not found or suspended."""

    def __init__(self, username: str):
        super().__init__(f"User '@{username}' not found, suspended, or private.", status_code=404)


# ── Progress callback type ────────────────────────────────────────
ProgressCallback = Callable[[int, str, str], Coroutine[Any, Any, None]]
# (step_number, status_key, human_message)


async def _noop_progress(step: int, status: str, message: str) -> None:
    """Default no-op progress callback."""
    pass


# ── Scweet Profile → MGTAB Profile Adapter ───────────────────────

def scweet_user_to_profile(user_data: dict) -> dict:
    """
    Map a Scweet v5 user-info dict to the exact 20-feature profile dict
    expected by our features.py pipeline.
    Scweet v5 user record fields (from DOCUMENTATION.md):
        user_id, username, name, description, location, created_at,
        followers_count, following_count, statuses_count, favourites_count,
        media_count, listed_count, verified, blue_verified, protected,
        profile_image_url, profile_banner_url, url
    Handles the Scweet → MGTAB field name mapping and injects
    dataset-mode defaults for 6 legacy fields not exposed by the
    modern Twitter/X internal API.
    """
    def _get(key: str, *alt_keys: str, default=None):
        """Try multiple keys in the dict, return the first found."""
        val = user_data.get(key)
        if val is not None:
            return val
        for k in alt_keys:
            val = user_data.get(k)
            if val is not None:
                return val
        return default

    # Scweet v5 uses "following_count"; map it to our "friends_count"
    # Also handle "blue_verified" as an alias for verified
    is_verified = bool(
        _get("verified", default=False)
        or _get("blue_verified", default=False)
    )

    # Scweet returns "profile_image_url"; check if it's the default egg/avatar
    profile_img = str(_get("profile_image_url", default="") or "")
    is_default_image = (
        "default_profile" in profile_img.lower()
        or not profile_img
    )

    return {
        # ── Available from Scweet get_user_info ──────────────
        "followers_count":  int(_get("followers_count", default=0) or 0),
        "friends_count":    int(_get("following_count", "friends_count", default=0) or 0),
        "listed_count":     int(_get("listed_count", default=0) or 0),
        "statuses_count":   int(_get("statuses_count", default=0) or 0),
        "favourites_count": int(_get("favourites_count", default=0) or 0),
        "name":             str(_get("name", default="") or ""),
        "screen_name":      str(_get("username", "screen_name", default="") or ""),
        "description":      str(_get("description", default="") or ""),
        "created_at":       _get("created_at", default=None),
        "default_profile":       False,  # Not exposed by Scweet; use safe default
        "default_profile_image": is_default_image,
        "verified":              is_verified,

        # ── Derived ──────────────────────────────────────────
        "has_url": bool(_get("url", default=None)),

        # ── Legacy fields NOT in modern API — dataset-mode defaults ──
        # These 6 fields were deprecated in Twitter API v2 / internal API.
        # We inject the statistically modal values from the MGTAB dataset
        # so the feature vector remains a valid 788-dim input.
        "geo_enabled":                       False,
        "profile_use_background_image":      True,
        "default_profile_background_color":  False,
        "default_profile_sidebar_fill_color":  False,
        "default_profile_sidebar_border_color": False,
        "profile_background_image_url":      False,
    }


def _extract_tweet_texts(tweets: list) -> list[str]:
    """Extract text from a list of Scweet tweet dicts.
    Scweet v5 tweet record has: tweet_id, timestamp, user, text,
    likes, retweets, comments, tweet_url, media, embedded_text, raw
    """
    texts = []
    for tweet in tweets:
        if isinstance(tweet, dict):
            text = tweet.get("text") or tweet.get("full_text") or ""
        else:
            text = getattr(tweet, "text", None) or getattr(tweet, "full_text", None) or ""
        if text and text.strip():
            texts.append(text.strip())
    return texts


def _extract_urls_from_tweets(tweets: list) -> set[str]:
    """Extract all URLs from tweet dicts for co-occurrence matching.
    Uses the raw GraphQL payload when available for structured URL entities,
    with a regex fallback on the tweet text.
    """
    urls = set()
    for tweet in tweets:
        if isinstance(tweet, dict):
            # Try structured entities from the raw payload
            raw = tweet.get("raw", {})
            legacy = raw.get("legacy", {}) if isinstance(raw, dict) else {}
            entities = legacy.get("entities", {}) if isinstance(legacy, dict) else {}
            tweet_urls = entities.get("urls", [])

            for url_obj in tweet_urls:
                if isinstance(url_obj, dict):
                    u = url_obj.get("expanded_url") or url_obj.get("url") or ""
                    if u:
                        urls.add(u.lower())

            # Fallback: extract URLs from text via regex
            text = tweet.get("text", "") or ""
            for match in re.findall(r"https?://\S+", text):
                urls.add(match.lower())
        else:
            text = getattr(tweet, "text", "") or ""
            for match in re.findall(r"https?://\S+", text):
                urls.add(match.lower())
    return urls


def _extract_hashtags_from_tweets(tweets: list) -> set[str]:
    """Extract all hashtags from tweet dicts for co-occurrence matching.
    Uses the raw GraphQL payload when available, with regex fallback.
    """
    hashtags = set()
    for tweet in tweets:
        if isinstance(tweet, dict):
            # Try structured entities from the raw payload
            raw = tweet.get("raw", {})
            legacy = raw.get("legacy", {}) if isinstance(raw, dict) else {}
            entities = legacy.get("entities", {}) if isinstance(legacy, dict) else {}
            tweet_hashtags = entities.get("hashtags", [])

            for tag in tweet_hashtags:
                if isinstance(tag, dict):
                    t = tag.get("text") or tag.get("tag") or ""
                    if t:
                        hashtags.add(t.lower())
                elif isinstance(tag, str):
                    hashtags.add(tag.lower().strip("#"))

            # Fallback: extract from text via regex
            text = tweet.get("text", "") or ""
            for match in re.findall(r"#(\w+)", text):
                hashtags.add(match.lower())
        else:
            text = getattr(tweet, "text", "") or ""
            for match in re.findall(r"#(\w+)", text):
                hashtags.add(match.lower())
    return hashtags


def _extract_mentions_from_tweets(tweets: list) -> set[str]:
    """Extract @mentioned usernames from tweet text."""
    mentions = set()
    for tweet in tweets:
        if isinstance(tweet, dict):
            text = tweet.get("text") or ""
        else:
            text = getattr(tweet, "text", "") or ""
        for match in re.findall(r"@(\w{1,15})", text):
            mentions.add(match.lower())
    return mentions


def _extract_reply_usernames(tweets: list) -> set[str]:
    """Extract usernames this user replied to from tweet dicts.
    Scweet v5 tweet records include a 'raw' field containing the full
    GraphQL payload, which has legacy.in_reply_to_screen_name.
    """
    reply_users = set()
    for tweet in tweets:
        if isinstance(tweet, dict):
            # Check the raw GraphQL payload for in_reply_to_screen_name
            raw = tweet.get("raw", {})
            legacy = raw.get("legacy", {}) if isinstance(raw, dict) else {}

            reply_to = None
            if isinstance(legacy, dict):
                reply_to = legacy.get("in_reply_to_screen_name")

            if reply_to and isinstance(reply_to, str):
                reply_users.add(reply_to.lower())
            else:
                # Fallback: check if tweet text starts with @mention (reply pattern)
                text = tweet.get("text") or ""
                if text.startswith("@"):
                    match = re.match(r"@(\w{1,15})", text)
                    if match:
                        reply_users.add(match.group(1).lower())
        else:
            reply_to = getattr(tweet, "in_reply_to_screen_name", None)
            if reply_to:
                reply_users.add(str(reply_to).lower())
    return reply_users


def _extract_quoted_usernames(tweets: list) -> set[str]:
    """Extract usernames of users who were quoted.
    Scweet v5 provides an 'embedded_text' field for quoted/retweeted tweets,
    and the raw payload often contains quoted_status_permalink or similar.
    """
    quoted = set()
    for tweet in tweets:
        if isinstance(tweet, dict):
            raw = tweet.get("raw", {})
            legacy = raw.get("legacy", {}) if isinstance(raw, dict) else {}

            # Check for quoted tweet data in raw payload
            if isinstance(legacy, dict) and legacy.get("is_quote_status"):
                # Try to extract from quoted_status_permalink
                permalink = legacy.get("quoted_status_permalink", {})
                if isinstance(permalink, dict):
                    expanded = permalink.get("expanded", "")
                    quote_match = re.search(r"(?:twitter\.com|x\.com)/(\w+)/status/", str(expanded))
                    if quote_match:
                        quoted.add(quote_match.group(1).lower())

            # Also try to extract from the tweet text (twitter.com/user/status/ pattern)
            text = tweet.get("text") or ""
            quote_match = re.search(r"(?:twitter\.com|x\.com)/(\w+)/status/", text)
            if quote_match:
                quoted.add(quote_match.group(1).lower())
    return quoted


# ── Rate-limit aware tweet fetcher ────────────────────────────────

def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit / HTTP 429 error.
    Checks the Scweet exception hierarchy first, then falls back to
    pattern-matching on the exception message for resilience against
    library version changes.
    """
    # 1. Check Scweet's typed exceptions
    try:
        from Scweet import RateLimitError, AccountPoolExhausted
        if isinstance(exc, (RateLimitError, AccountPoolExhausted)):
            return True
    except ImportError:
        pass

    # 2. Fallback: pattern match on the error string
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg or "too many" in msg:
        return True

    return False


# ── Scweet Scraper Singleton ──────────────────────────────────────

class ScweetScraper:
    """
    Async Twitter/X scraper using Scweet v5.
    Authenticates via auth_token cookie (no passwords needed).
    Supports optional HTTP/HTTPS proxy via PROXY_URL env var.
    """

    def __init__(self):
        self._client = None
        self._auth_token = TWITTER_AUTH_TOKEN

    async def _ensure_client(self) -> None:
        """
        Lazily initialize the Scweet client.
        Uses auth_token extracted from browser cookies.
        Proxy configuration: if PROXY_URL is set, the ScweetConfig is
        initialized with proxies={"http": PROXY_URL, "https": PROXY_URL}
        so all Scweet HTTP requests are tunnelled through the proxy.
        Runs Scweet init in a thread to avoid blocking the event loop
        (Scweet provisions accounts into SQLite on first init).
        """
        if self._client is not None:
            return

        if not self._auth_token:
            raise ScraperAuthError(
                "TWITTER_AUTH_TOKEN not configured. "
                "Log into x.com → F12 → Application → Cookies → copy auth_token value, "
                "then set it in backend/.env"
            )

        try:
            from Scweet import Scweet, ScweetConfig, ScweetDB

            # Initialize in a thread since Scweet provisioning uses SQLite I/O
            def _init_client():
                # Build config kwargs — include proxy if provided
                config_kwargs = {
                    "daily_requests_limit": 500,
                    "daily_tweets_limit": 5000,
                    "min_delay_s": 1.0,
                }

                if PROXY_URL:
                    config_kwargs["proxies"] = {
                        "http": PROXY_URL,
                        "https": PROXY_URL,
                    }
                    logger.info(f"Scweet: Proxy configured → {PROXY_URL[:30]}...")

                config = ScweetConfig(**config_kwargs)

                client = Scweet(
                    auth_token=self._auth_token,
                    manifest_scrape_on_init=True,
                    config=config,
                )

                # Reset daily counters to clear any previous lockout
                try:
                    db = ScweetDB(config.db_path)
                    db.reset_daily_counters()
                    logger.info("Scweet: Daily counters reset.")
                except Exception as db_err:
                    logger.warning(f"Scweet: Could not reset daily counters: {db_err}")
                return client

            self._client = await asyncio.to_thread(_init_client)
            logger.info(
                f"Scweet: Client initialized with auth_token (v5, GraphQL mode). "
                f"Delay={SCRAPE_DELAY_SECONDS}s, Proxy={'YES' if PROXY_URL else 'NO'}"
            )
        except ImportError:
            raise ScraperAuthError(
                "Scweet is not installed. Run: pip install -U Scweet"
            )
        except Exception as e:
            logger.exception("Scweet initialization failed")
            raise ScraperAuthError(f"Scweet init failed: {str(e)}")

    async def _safe_delay(self) -> None:
        """Sleep between API calls to respect rate limits.
        Uses the SCRAPE_DELAY_SECONDS env var (default 3.0s).
        """
        await asyncio.sleep(SCRAPE_DELAY_SECONDS)

    def _handle_error(self, e: Exception, context: str) -> None:
        """Convert Scweet exceptions to our ScraperError types."""
        error_msg = str(e).lower()
        error_type = type(e).__name__
        logger.warning(f"Scweet error during {context}: [{error_type}] {e}")

        # Match Scweet v5 exception hierarchy:
        #   ScweetError
        #     AccountPoolExhausted
        #     EngineError
        #       RunFailed
        #         RateLimitError
        #         AuthError
        #         NetworkError
        #         ProxyError
        try:
            from Scweet import (
                AccountPoolExhausted,
                RateLimitError,
                AuthError,
                NetworkError,
            )

            if isinstance(e, RateLimitError):
                raise ScraperRateLimitError()
            elif isinstance(e, AuthError):
                self._client = None  # Force re-init next time
                raise ScraperAuthError(f"Auth error during {context}: {e}")
            elif isinstance(e, AccountPoolExhausted):
                raise ScraperRateLimitError(
                    "All Scweet accounts exhausted / cooling down. Try again later."
                )
            elif isinstance(e, NetworkError):
                raise ScraperError(f"Network error during {context}: {e}")
        except ImportError:
            pass

        # Fallback: pattern match on error message
        if "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
            raise ScraperRateLimitError()
        elif "not found" in error_msg or "404" in error_msg or "does not exist" in error_msg:
            raise ScraperUserNotFoundError(context)
        elif "auth" in error_msg or "401" in error_msg or "unauthorized" in error_msg:
            self._client = None  # Force re-init next time
            raise ScraperAuthError(f"Auth error during {context}: {e}")
        elif "suspended" in error_msg or "403" in error_msg:
            raise ScraperError(f"Account suspended/forbidden: {e}", status_code=403)
        else:
            raise ScraperError(f"Scraping error during {context}: {e}")

    # ── Core Scrape Functions ─────────────────────────────────────

    async def scrape_user_profile(self, username: str) -> dict:
        """Fetch user profile info by screen name.
        Uses Scweet's aget_user_info([username]) which returns a list
        of user record dicts.
        """
        await self._ensure_client()
        try:
            profiles = await self._client.aget_user_info([username])
            if not profiles:
                raise ScraperUserNotFoundError(username)
            # Return the first (and only) profile
            profile = profiles[0] if isinstance(profiles, list) else profiles
            return profile if isinstance(profile, dict) else {}
        except (ScraperError, ScraperUserNotFoundError):
            raise
        except Exception as e:
            self._handle_error(e, username)

    async def scrape_user_tweets(self, username: str, count: int = 20) -> list:
        """Fetch recent tweets for a username.
        Uses Scweet's aget_profile_tweets([username], limit=count).
        Returns a list of tweet record dicts.
        RATE-LIMIT RESILIENT: If a RateLimitError or HTTP 429 is caught,
        logs a warning and returns an empty list instead of crashing.
        This ensures the node still enters the graph with profile features
        even when the content pillar is starved.
        """
        await self._ensure_client()
        try:
            tweets = await self._client.aget_profile_tweets([username], limit=count)
            return list(tweets) if tweets else []
        except Exception as e:
            if _is_rate_limit_error(e):
                logger.warning(
                    f"⚠ Rate-limited fetching tweets for @{username} "
                    f"(HTTP 429). Returning empty tweet list — node will "
                    f"enter the graph with profile features only."
                )
            else:
                logger.warning(f"Could not fetch tweets for @{username}: {e}")
            return []

    async def scrape_followers(self, username: str, count: int = 10) -> list:
        """Fetch followers for a username.
        Uses Scweet's aget_followers([username], limit=count).
        Returns list of user record dicts.
        """
        await self._ensure_client()
        try:
            followers = await self._client.aget_followers([username], limit=count)
            return list(followers)[:count] if followers else []
        except Exception as e:
            logger.warning(f"Could not fetch followers for @{username}: {e}")
            return []

    async def scrape_following(self, username: str, count: int = 10) -> list:
        """Fetch following for a username.
        Uses Scweet's aget_following([username], limit=count).
        Returns list of user record dicts.
        """
        await self._ensure_client()
        try:
            following = await self._client.aget_following([username], limit=count)
            return list(following)[:count] if following else []
        except Exception as e:
            logger.warning(f"Could not fetch following for @{username}: {e}")
            return []

    # ── Ego-Graph Expansion ───────────────────────────────────────

    async def scrape_ego_graph(
        self,
        username: str,
        progress: ProgressCallback = _noop_progress,
    ) -> dict:
        """
        Full ego-graph scrape pipeline.
        1. Authenticate + scrape target user profile + 20 recent tweets
        2. Discover neighbors via 5 relation types (up to 10 each)
        3. Fetch real profile + 5 tweets for each unique neighbor
        4. Build relation edges (including URL/hashtag co-occurrence)
        Step 5 (RGCN inference) is handled by main.py's SSE endpoint.
        SSE Progress Steps (must match React stepper UI):
            step 1 → "scraping_profile"   ("Authenticating")
            step 2 → "fetching_network"   ("Fetching Network")
            step 3 → "enriching_neighbors" ("Enriching Neighbors")
            step 4 → "building_graph"     ("Building Graph")
            step 5 → (emitted by main.py) ("Running RGCN")
        Rate-Limit Resilience:
            Tweet fetching for both the target and each neighbor is wrapped
            in try-except blocks that catch RateLimitError / HTTP 429.
            If rate-limited, the node enters the graph with profile features
            and an empty tweet list, rather than crashing the pipeline.
        Args:
            username: Twitter handle (with or without @)
            progress: async callback for SSE progress streaming
        Returns:
            tuple of (request_data dict, scrape_meta dict)
        """
        # Clean the username
        username = username.strip().lstrip("@")
        if not username:
            raise ScraperError("Username cannot be empty.", status_code=400)

        # ── Step 1: Authenticate + Scrape target ─────────────────
        await progress(1, "scraping_profile", f"Authenticating & scraping profile for @{username}...")
        await self._ensure_client()

        target_profile_raw = await self.scrape_user_profile(username)

        # Check if the account is protected
        if target_profile_raw.get("protected"):
            raise ScraperError(
                f"@{username} has a private/protected account. Cannot scrape.",
                status_code=403,
            )

        await self._safe_delay()

        # Rate-limit resilient tweet fetch for the TARGET user
        target_tweets_raw = await self.scrape_user_tweets(username, MAX_TWEETS_TARGET)
        target_tweets = _extract_tweet_texts(target_tweets_raw)
        target_profile = scweet_user_to_profile(target_profile_raw)
        target_urls = _extract_urls_from_tweets(target_tweets_raw)
        target_hashtags = _extract_hashtags_from_tweets(target_tweets_raw)

        logger.info(f"Target @{username}: {len(target_tweets)} tweets scraped.")

        # ── Step 2: Discover neighbors via relations ──────────────
        await progress(2, "fetching_network", "Discovering social network (followers, friends, interactions)...")

        # screen_name_lower → { "profile_raw": dict, "relations": set, "screen_name": str }
        discovered: dict[str, dict] = {}

        def _register(user_info: Any, relation: str) -> None:
            """Register a discovered neighbor."""
            if isinstance(user_info, dict):
                # Scweet v5 uses "username" as the field name
                sn = (user_info.get("username") or user_info.get("screen_name") or "").lower()
            else:
                sn = str(getattr(user_info, "username", "") or getattr(user_info, "screen_name", "")).lower()

            if not sn or sn == username.lower():
                return  # skip self-loops

            if sn not in discovered:
                discovered[sn] = {
                    "profile_raw": user_info if isinstance(user_info, dict) else {},
                    "relations": set(),
                    "screen_name": sn,
                }
            discovered[sn]["relations"].add(relation)

        # 2a. Followers (up to MAX_NEIGHBORS_PER_RELATION)
        try:
            await self._safe_delay()
            followers = await self.scrape_followers(username, MAX_NEIGHBORS_PER_RELATION)
            for f in followers:
                _register(f, "follower")
            logger.info(f"  Followers scraped: {len(followers)}")
        except Exception as e:
            logger.warning(f"  Followers scrape failed: {e}")

        # 2b. Friends / following (up to MAX_NEIGHBORS_PER_RELATION)
        try:
            await self._safe_delay()
            following = await self.scrape_following(username, MAX_NEIGHBORS_PER_RELATION)
            for f in following:
                _register(f, "friend")
            logger.info(f"  Following scraped: {len(following)}")
        except Exception as e:
            logger.warning(f"  Following scrape failed: {e}")

        # 2c. Mentions — parsed from target's tweets
        mentioned_usernames = _extract_mentions_from_tweets(target_tweets_raw)
        mentioned_usernames.discard(username.lower())
        for sn in list(mentioned_usernames)[:MAX_NEIGHBORS_PER_RELATION]:
            if sn in discovered:
                discovered[sn]["relations"].add("mention")
            else:
                discovered[sn] = {
                    "profile_raw": {},
                    "relations": {"mention"},
                    "screen_name": sn,
                }

        # 2d. Replies — parsed from target's tweets
        reply_usernames = _extract_reply_usernames(target_tweets_raw)
        reply_usernames.discard(username.lower())
        for sn in list(reply_usernames)[:MAX_NEIGHBORS_PER_RELATION]:
            if sn in discovered:
                discovered[sn]["relations"].add("reply")
            else:
                discovered[sn] = {
                    "profile_raw": {},
                    "relations": {"reply"},
                    "screen_name": sn,
                }

        # 2e. Quotes — parsed from target's tweets
        quoted_usernames = _extract_quoted_usernames(target_tweets_raw)
        quoted_usernames.discard(username.lower())
        for sn in list(quoted_usernames)[:MAX_NEIGHBORS_PER_RELATION]:
            if sn in discovered:
                discovered[sn]["relations"].add("quoted")
            else:
                discovered[sn] = {
                    "profile_raw": {},
                    "relations": {"quoted"},
                    "screen_name": sn,
                }

        logger.info(f"  Total unique neighbors discovered: {len(discovered)}")

        # ── Step 3: Enrich neighbors with real data ───────────────
        await progress(3, "enriching_neighbors", f"Fetching data for {len(discovered)} neighbors...")

        neighbors_data: list[dict] = []
        neighbor_urls_map: dict[str, set] = {}    # screen_name → set of urls
        neighbor_hashtags_map: dict[str, set] = {}  # screen_name → set of hashtags

        for sn, info in discovered.items():
            # Fetch profile if we don't have it yet (mentions/replies/quotes)
            if not info["profile_raw"]:
                try:
                    await self._safe_delay()
                    info["profile_raw"] = await self.scrape_user_profile(sn)
                except Exception as e:
                    logger.warning(f"    Could not fetch profile for @{sn}: {e}")
                    # Use minimal profile data so the node still enters the graph
                    info["profile_raw"] = {"username": sn}

            # Fetch tweets for this neighbor — RATE-LIMIT RESILIENT
            # If rate-limited, n_tweets_raw stays [], and the node still
            # enters the graph with profile-only features.
            n_tweets_raw: list = []
            try:
                await self._safe_delay()
                n_tweets_raw = await self.scrape_user_tweets(sn, MAX_TWEETS_NEIGHBOR)
            except Exception as e:
                if _is_rate_limit_error(e):
                    logger.warning(
                        f"    ⚠ Rate-limited on tweets for neighbor @{sn}. "
                        f"Node enters graph with profile features only."
                    )
                else:
                    logger.warning(f"    Could not fetch tweets for @{sn}: {e}")
                n_tweets_raw = []

            n_tweet_texts = _extract_tweet_texts(n_tweets_raw)
            n_profile = scweet_user_to_profile(info["profile_raw"])

            # Track URLs and hashtags for co-occurrence edges
            neighbor_urls_map[sn] = _extract_urls_from_tweets(n_tweets_raw)
            neighbor_hashtags_map[sn] = _extract_hashtags_from_tweets(n_tweets_raw)

            neighbors_data.append({
                "id": sn,
                "profile": n_profile,
                "tweets": n_tweet_texts,
            })

            logger.info(f"    Enriched @{sn}: {len(n_tweet_texts)} tweets")

        # ── Step 4: Build relations list ──────────────────────────
        await progress(4, "building_graph", "Building relation edges and encoding features...")

        relations: list[dict] = []

        for sn, info in discovered.items():
            for rel_type in info["relations"]:
                if rel_type == "follower":
                    # follower: neighbor follows target → source=neighbor, target=target
                    relations.append({
                        "source": sn,
                        "target": "__target__",
                        "relation": "follower",
                    })
                else:
                    # friend, mention, reply, quoted: target → neighbor
                    relations.append({
                        "source": "__target__",
                        "target": sn,
                        "relation": rel_type,
                    })

        # Opportunistic URL co-occurrence edges
        for sn, n_urls in neighbor_urls_map.items():
            if target_urls & n_urls:  # intersection
                relations.append({
                    "source": "__target__",
                    "target": sn,
                    "relation": "url",
                })

        # Opportunistic hashtag co-occurrence edges
        for sn, n_hashtags in neighbor_hashtags_map.items():
            if target_hashtags & n_hashtags:  # intersection
                relations.append({
                    "source": "__target__",
                    "target": sn,
                    "relation": "hashtag",
                })

        logger.info(
            f"Ego-graph for @{username}: "
            f"{len(neighbors_data)} neighbors, {len(relations)} relations"
        )

        # ── Assemble the request dict ─────────────────────────────
        request_data = {
            "target": {
                "profile": target_profile,
                "tweets": target_tweets,
            },
            "neighbors": neighbors_data,
            "relations": relations,
        }

        # Include scrape metadata for the frontend summary card
        relation_counts: dict[str, int] = {}
        for r in relations:
            rt = r["relation"]
            relation_counts[rt] = relation_counts.get(rt, 0) + 1

        scrape_meta = {
            "username": username,
            "display_name": target_profile["name"],
            "followers_count": target_profile["followers_count"],
            "friends_count": target_profile["friends_count"],
            "tweets_scraped": len(target_tweets),
            "neighbors_found": len(neighbors_data),
            "total_relations": len(relations),
            "relation_breakdown": relation_counts,
        }

        return request_data, scrape_meta


# ── Module-level singleton ────────────────────────────────────────
_scraper: Optional[ScweetScraper] = None


def get_scraper() -> ScweetScraper:
    """Get or create the global ScweetScraper singleton."""
    global _scraper
    if _scraper is None:
        _scraper = ScweetScraper()
    return _scraper

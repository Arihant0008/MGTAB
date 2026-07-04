"""
FastAPI application — MGTAB Bot Detector API.
Queue-driven architecture with SQLite persistence and UID-based rate limiting.

Endpoints:
    POST /api/v1/predict            — Submit a scrape job (rate limited, returns 202)
    GET  /api/v1/jobs/{job_id}      — Poll job status and results
    POST /predict/user              — Manual mode (synchronous, backward compat)
    GET  /model/info                — Model metadata
    GET  /health                    — Health check
    GET  /features/schema           — Feature definitions for frontend
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import firebase_admin
from firebase_admin import credentials as fb_credentials

from .config import CORS_ORIGINS, RELATION_MAP, NUM_FEATURES, HIDDEN_DIM, NUM_CLASSES, NUM_RELATIONS
from .inference import InferenceEngine
from .cache import cache_get, cache_set, cache_delete
from .database import init_db, check_rate_limit, create_job, update_job, get_job
from .security import (
    SecurityHeadersMiddleware,
    PredictJobRequest,
    get_current_uid,
    verify_bearer_token,
)
from .scraper import (
    ScraperAuthError,
    ScraperError,
    ScraperRateLimitError,
    ScraperUserNotFoundError,
    get_scraper,
)

# Load .env file for Twitter credentials
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global inference engine ───────────────────────────────────────
engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, initialize Firebase Admin & SQLite, clean up on shutdown."""
    global engine
    logger.info("Starting MGTAB Bot Detector API...")

    # ── Firebase Admin Initialization ──────────────────────────────
    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if sa_json:
        try:
            sa_dict = json.loads(sa_json)
            cred = fb_credentials.Certificate(sa_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized from FIREBASE_SERVICE_ACCOUNT.")
        except Exception as e:
            logger.warning(f"Firebase Admin init failed: {e}. Auth verification disabled.")
    else:
        logger.warning(
            "FIREBASE_SERVICE_ACCOUNT not set — auth verification disabled. "
            "Set this secret in your Hugging Face Space settings."
        )

    # ── SQLite Initialization (with boot-up recovery sweeper) ─────
    init_db()

    # ── Inference Engine ──────────────────────────────────────────
    engine = InferenceEngine()
    logger.info("Inference engine ready.")

    # Pre-warm the scraper (lazy login happens on first request)
    _ = get_scraper()
    logger.info("Scweet scraper initialized (auth deferred to first use).")

    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="MGTAB Bot Detector API",
    description="Classify Twitter/X accounts as bot or human using RGCN on MGTAB features.",
    version="3.0.0",
    lifespan=lifespan,
)

# ── Security Headers Middleware ───────────────────────────────────
app.add_middleware(SecurityHeadersMiddleware)

# ── CORS — dynamic production origin + local dev origins ─────────
_cors_origins = list(CORS_ORIGINS)  # from config.py (localhost dev ports)
_production_origin = os.getenv("CORS_PRODUCTION_ORIGIN", "")
if _production_origin:
    _cors_origins.append(_production_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins if _cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)


# ── Pydantic Models ──────────────────────────────────────────────

class ProfileData(BaseModel):
    """Raw profile fields from the frontend."""
    followers_count: float = 0
    friends_count: float = 0
    listed_count: float = 0
    statuses_count: float = 0
    favourites_count: float = 0
    name: str = ""
    screen_name: str = ""
    description: str = ""
    created_at: Optional[str] = None
    default_profile: bool = False
    default_profile_image: bool = False
    verified: bool = False
    has_url: bool = False
    geo_enabled: bool = False
    profile_use_background_image: bool = True
    default_profile_background_color: bool = False
    default_profile_sidebar_fill_color: bool = False
    default_profile_sidebar_border_color: bool = False
    profile_background_image_url: bool = False


class NeighborData(BaseModel):
    """Data for a neighbor node."""
    id: str
    profile: Optional[ProfileData] = None
    tweets: list[str] = Field(default_factory=list)


class RelationData(BaseModel):
    """A single edge/relation."""
    source: str
    target: str
    relation: str


class TargetData(BaseModel):
    """Target user data."""
    profile: ProfileData
    tweets: list[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    """Full prediction request body (manual mode)."""
    target: TargetData
    neighbors: list[NeighborData] = Field(default_factory=list)
    relations: list[RelationData] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "target": {
                        "profile": {
                            "followers_count": 150,
                            "friends_count": 200,
                            "listed_count": 5,
                            "statuses_count": 3000,
                            "favourites_count": 500,
                            "name": "John Doe",
                            "screen_name": "johndoe",
                            "description": "Just a regular user",
                            "created_at": "2018-05-15T00:00:00Z",
                            "default_profile": False,
                            "default_profile_image": False,
                            "verified": False,
                            "has_url": True,
                            "geo_enabled": True,
                        },
                        "tweets": [
                            "Great weather today!",
                            "Just finished reading a great book.",
                            "Looking forward to the weekend."
                        ],
                    },
                    "neighbors": [],
                    "relations": [],
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Prediction response."""
    label_pred: str
    prob_human: float
    prob_bot: float
    confidence: float
    graph_info: dict
    quality_warning: Optional[str] = None  # Set when graph coverage is low


# ══════════════════════════════════════════════════════════════════
#  QUEUE-DRIVEN ENDPOINTS (v3)
# ══════════════════════════════════════════════════════════════════

# ── Background Task Runner ────────────────────────────────────────

def run_scrape_job(job_id: str, handle: str, refresh: bool):
    """
    Background task that runs the full scrape → inference pipeline.
    Updates job status in SQLite throughout execution.
    Wrapped in try/except to guarantee no deadlocked jobs.
    """
    try:
        update_job(job_id, status="processing")

        # ── Cache Check ───────────────────────────────────────────
        if not refresh:
            cached = cache_get(handle)
            if cached is not None:
                update_job(
                    job_id,
                    status="completed",
                    result=json.dumps(cached),
                    progress=json.dumps({
                        "step": 5, "status_key": "complete", "message": "Retrieved from cache"
                    }),
                )
                return
        else:
            cache_delete(handle)

        # ── Scrape Ego-Graph ──────────────────────────────────────
        scraper = get_scraper()

        # Synchronous progress callback that writes to SQLite
        def sync_progress(step: int, status_key: str, message: str):
            update_job(
                job_id,
                progress=json.dumps({
                    "step": step, "status_key": status_key, "message": message
                }),
            )

        # Create a new event loop for this thread (BackgroundTasks runs in threadpool)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Wrap async scraper with sync progress bridge
            async def run_scraper():
                async def async_progress(step, status_key, message):
                    sync_progress(step, status_key, message)
                return await scraper.scrape_ego_graph(handle, progress=async_progress)

            request_data, scrape_meta = loop.run_until_complete(run_scraper())
        finally:
            loop.close()

        # ── Run RGCN Inference ────────────────────────────────────
        sync_progress(5, "running_rgcn", "Running RGCN inference...")

        result = engine.predict_from_request(request_data)
        result["graph_info"]["scrape_meta"] = scrape_meta

        # Remove high-follower calibration warning (technical, not user-facing)
        if result.get("quality_warning") and "followers" in result["quality_warning"]:
            result["quality_warning"] = None

        # ── Cache Result ──────────────────────────────────────────
        cache_set(handle, result)

        # ── Mark Complete ─────────────────────────────────────────
        update_job(
            job_id,
            status="completed",
            result=json.dumps(result),
            progress=json.dumps({
                "step": 5, "status_key": "complete", "message": "Analysis complete"
            }),
        )
        logger.info(f"Job {job_id} completed: {handle} → {result['label_pred']}")

    except ScraperUserNotFoundError as e:
        update_job(job_id, status="failed", error=f"User not found: {e}")
        logger.warning(f"Job {job_id} failed (user not found): {handle}")
    except ScraperRateLimitError as e:
        update_job(job_id, status="failed", error=f"Twitter rate limit hit: {e}")
        logger.warning(f"Job {job_id} failed (rate limit): {handle}")
    except ScraperAuthError as e:
        update_job(job_id, status="failed", error=f"Twitter authentication error: {e}")
        logger.warning(f"Job {job_id} failed (auth): {handle}")
    except ScraperError as e:
        update_job(job_id, status="failed", error=f"Scraping error: {e}")
        logger.warning(f"Job {job_id} failed (scraper): {handle}")
    except Exception as e:
        update_job(job_id, status="failed", error=f"Unexpected error: {str(e)}")
        logger.exception(f"Job {job_id} failed unexpectedly: {handle}")


# ── 1. Submit Prediction Job ─────────────────────────────────────

@app.post("/api/v1/predict", status_code=202)
async def submit_prediction(
    body: PredictJobRequest,
    background_tasks: BackgroundTasks,
    uid: str = Depends(get_current_uid),
):
    """
    Submit a new bot detection job. Returns 202 Accepted with job_id.

    Flow:
    1. Validate input (PredictJobRequest — 15-char alphanumeric regex)
    2. Extract Firebase UID from Bearer token
    3. Check 24-hour rate limit → 429 if within cooldown
    4. Check Redis cache → instant completion if cached
    5. Create job record → enqueue background task
    6. Return job_id for polling
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Rate limit check (raises 429 if within 24h window)
    check_rate_limit(uid)

    handle = body.target_handle.strip().lower()

    # Check cache for instant result (skip queue entirely)
    if not body.refresh:
        cached = cache_get(handle)
        if cached is not None:
            job_id = create_job(uid, handle)
            update_job(
                job_id,
                status="completed",
                result=json.dumps(cached),
                progress=json.dumps({
                    "step": 5, "status_key": "complete", "message": "Retrieved from cache"
                }),
            )
            return {
                "job_id": job_id,
                "status": "completed",
                "poll_url": f"/api/v1/jobs/{job_id}",
                "from_cache": True,
            }

    # Create job and enqueue background task
    job_id = create_job(uid, handle)
    background_tasks.add_task(run_scrape_job, job_id, handle, body.refresh)

    return {
        "job_id": job_id,
        "status": "queued",
        "poll_url": f"/api/v1/jobs/{job_id}",
    }


# ── 2. Poll Job Status ───────────────────────────────────────────

@app.get("/api/v1/jobs/{job_id}")
async def poll_job_status(
    job_id: str,
    _uid: str = Depends(get_current_uid),
):
    """
    Poll the status of a submitted job.
    Returns current status, progress step, result (if completed), or error (if failed).
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "target_handle": job["target_handle"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
        "created_at": job["created_at"],
    }

    return response


# ══════════════════════════════════════════════════════════════════
#  LEGACY / STANDARD ENDPOINTS
# ══════════════════════════════════════════════════════════════════

# ── Manual Mode (backward compatible) ────────────────────────────

@app.post("/predict/user", response_model=PredictResponse)
async def predict_user(request: PredictRequest, _claims=Depends(verify_bearer_token)):
    """
    Classify a Twitter/X account as bot or human.
    
    Accepts raw profile data, tweets, and optional neighbor/relation data.
    Returns classification label with probabilities.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        request_dict = request.model_dump()
        # Convert nested profile models to plain dicts
        result = engine.predict_from_request(request_dict)
        return PredictResponse(**result)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── Metadata & Health ─────────────────────────────────────────────

@app.get("/model/info")
async def model_info(_claims=Depends(verify_bearer_token)):
    """Return model metadata."""
    return {
        "model": "RGCN",
        "architecture": f"RGCNConv({NUM_FEATURES}→{HIDDEN_DIM}→{NUM_CLASSES})",
        "num_relations": NUM_RELATIONS,
        "relation_types": list(RELATION_MAP.keys()),
        "test_accuracy": 0.8823,
        "bot_recall": 0.9029,
        "training_epochs": 200,
        "dataset": "MGTAB",
        "dataset_size": 10199,
        "feature_dim": NUM_FEATURES,
        "profile_features": 20,
        "tweet_features": 768,
        "tweet_encoder": "LaBSE",
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
    }


@app.get("/features/schema")
async def features_schema(_claims=Depends(verify_bearer_token)):
    """Return the 20 feature definitions for the frontend form."""
    return {
        "boolean_features": [
            {"key": "profile_use_background_image", "label": "Profile has background image", "index": 0},
            {"key": "default_profile", "label": "Default profile (not customized)", "index": 1},
            {"key": "verified", "label": "Verified account", "index": 2},
            {"key": "default_profile_image", "label": "Default profile image (egg/default avatar)", "index": 4},
            {"key": "geo_enabled", "label": "Geolocation enabled", "index": 8},
            {"key": "default_profile_background_color", "label": "Default background color", "index": 15},
            {"key": "default_profile_sidebar_fill_color", "label": "Default sidebar fill color", "index": 16},
            {"key": "default_profile_sidebar_border_color", "label": "Default sidebar border color", "index": 17},
            {"key": "has_url", "label": "Profile has URL", "index": 18},
            {"key": "profile_background_image_url", "label": "Profile background image has URL", "index": 19},
        ],
        "numerical_features": [
            {"key": "followers_count", "label": "Followers count", "index": 3},
            {"key": "listed_count", "label": "Listed count (public lists)", "index": 5},
            {"key": "statuses_count", "label": "Statuses count (tweets + retweets)", "index": 6},
            {"key": "friends_count", "label": "Friends count (following)", "index": 7},
            {"key": "favourites_count", "label": "Favourites count (likes)", "index": 9},
        ],
        "derived_features": [
            {"key": "created_at", "label": "Account creation date", "index": 10, "type": "date"},
            {"key": "screen_name_length", "label": "Screen name length", "index": 11, "derived_from": "screen_name"},
            {"key": "name_length", "label": "Name length", "index": 12, "derived_from": "name"},
            {"key": "description_length", "label": "Description length", "index": 13, "derived_from": "description"},
            {"key": "followers_friends_ratios", "label": "Followers/Friends ratio", "index": 14, "derived_from": "followers_count / friends_count"},
        ],
        "relation_types": list(RELATION_MAP.keys()),
    }

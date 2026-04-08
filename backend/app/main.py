"""
FastAPI application — MGTAB Bot Detector API.

Endpoints:
    POST /predict/user   — Classify a Twitter/X account
    GET  /model/info     — Model metadata
    GET  /health         — Health check
    GET  /features/schema — Feature definitions for frontend
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import CORS_ORIGINS, RELATION_MAP, NUM_FEATURES, HIDDEN_DIM, NUM_CLASSES, NUM_RELATIONS
from .inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global inference engine ───────────────────────────────────────────
engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, clean up on shutdown."""
    global engine
    logger.info("Starting MGTAB Bot Detector API...")
    engine = InferenceEngine()
    logger.info("Inference engine ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MGTAB Bot Detector API",
    description="Classify Twitter/X accounts as bot or human using RGCN on MGTAB features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Pydantic Models ──────────────────────────────────────────────────

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
    """Full prediction request body."""
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


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/predict/user", response_model=PredictResponse)
async def predict_user(request: PredictRequest):
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


@app.get("/model/info")
async def model_info():
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
async def features_schema():
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

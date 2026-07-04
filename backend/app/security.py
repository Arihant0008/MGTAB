"""
Application-layer security hardening.
- Strict input validation for target handles
- Firebase UID extraction dependency
- Security headers middleware (nosniff, frame deny)
"""

import logging
from typing import Optional

from fastapi import HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

import firebase_admin
from firebase_admin import auth as fb_auth

logger = logging.getLogger(__name__)


# ── Security Headers Middleware ───────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Append defense-in-depth headers to every outgoing response:
    - X-Content-Type-Options: nosniff — prevent MIME-type sniffing
    - X-Frame-Options: DENY — prevent clickjacking via iframes
    """
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response


# ── Input Validation Model ────────────────────────────────────────

class PredictJobRequest(BaseModel):
    """
    Strict input model for the job submission endpoint.
    Regex rejects all special characters — no command injection or XSS
    payloads reach the scraper. Max 15 chars matches Twitter/X handle limits.
    """
    target_handle: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_]{1,15}$",
        max_length=15,
        description="Twitter/X handle (alphanumeric and underscores only, max 15 chars)",
        examples=["elonmusk", "jack"],
    )
    refresh: bool = Field(
        default=False,
        description="If true, bypass cache and run a fresh scrape",
    )


# ── Firebase UID Extraction ───────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_uid(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    Extract and verify the Firebase UID from the Authorization: Bearer header.
    Returns the UID string for downstream rate limiting and job tracking.
    
    In dev mode (Firebase Admin not initialized), returns a placeholder UID.
    """
    if not firebase_admin._apps:
        # Dev mode — no verification, return placeholder
        logger.debug("Firebase Admin not initialized, using dev UID.")
        return "dev-user"

    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token.")

    try:
        decoded = fb_auth.verify_id_token(creds.credentials)
        return decoded["uid"]
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=401, detail="Invalid or expired authentication token."
        )


async def verify_bearer_token(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """
    Dependency for standard HTTP routes that need auth but not UID.
    Used by model/info, features/schema, etc.
    """
    if not firebase_admin._apps:
        return None  # Dev mode — no verification
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token.")
    try:
        decoded = fb_auth.verify_id_token(creds.credentials)
        return decoded
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=401, detail="Invalid or expired authentication token."
        )

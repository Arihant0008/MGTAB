"""
SQLite persistence layer for rate limiting and job tracking.
Uses WAL mode for concurrent read/write safety.
Designed for Hugging Face Spaces where container sleep cycles require local state.
"""

import json
import sqlite3
import threading
import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = "data.db"

# Thread-local storage for SQLite connections
_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """
    Return a thread-local SQLite connection with WAL mode and 30s busy timeout.
    Each thread gets its own connection to avoid locking issues.
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def init_db():
    """
    Create tables if they don't exist and run boot-up recovery sweeper.
    Called once on application startup.
    """
    conn = get_connection()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            uid TEXT PRIMARY KEY,
            last_request_time TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            uid TEXT NOT NULL,
            target_handle TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            progress TEXT,
            result TEXT,
            error TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_uid ON jobs(uid);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    """)

    conn.commit()
    logger.info("SQLite database initialized (WAL mode, 30s busy timeout).")

    # ── Boot-Up Recovery Sweeper ──────────────────────────────────
    # Mark any lingering queued/processing jobs as failed.
    # This resolves deadlocked rows from container recycles or worker crashes.
    now = _utcnow_iso()
    cursor = conn.execute(
        """
        UPDATE jobs
        SET status = 'failed',
            error = 'Worker process terminated unexpectedly. Please retry your request.',
            updated_at = ?
        WHERE status IN ('queued', 'processing')
        """,
        (now,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        logger.warning(
            f"Boot-up recovery: marked {cursor.rowcount} stale job(s) as failed."
        )


# ── Rate Limiting ─────────────────────────────────────────────────

def check_rate_limit(uid: str) -> None:
    """
    Enforce 24-hour (86,400s) global rate limit per UID.
    Raises HTTP 429 with retry_after_seconds if within the cooldown window.
    Updates last_request_time if the request is allowed.
    """
    from fastapi import HTTPException

    conn = get_connection()
    row = conn.execute(
        "SELECT last_request_time FROM rate_limits WHERE uid = ?", (uid,)
    ).fetchone()

    if row:
        last_time = datetime.fromisoformat(row["last_request_time"])
        # Ensure timezone-aware comparison
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = (now - last_time).total_seconds()

        if delta < 86400:
            remaining = int(86400 - delta)
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            raise HTTPException(
                status_code=429,
                detail={
                    "message": f"Rate limit exceeded. Try again in {hours}h {minutes}m.",
                    "retry_after_seconds": remaining,
                },
            )

    # Allow request — upsert last_request_time
    conn.execute(
        """
        INSERT INTO rate_limits (uid, last_request_time) VALUES (?, ?)
        ON CONFLICT(uid) DO UPDATE SET last_request_time = excluded.last_request_time
        """,
        (uid, _utcnow_iso()),
    )
    conn.commit()


# ── Job CRUD ──────────────────────────────────────────────────────

def create_job(uid: str, target_handle: str) -> str:
    """Insert a new job row with status='queued'. Returns the job_id."""
    job_id = str(uuid.uuid4())
    now = _utcnow_iso()
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO jobs (job_id, uid, target_handle, status, created_at, updated_at)
        VALUES (?, ?, ?, 'queued', ?, ?)
        """,
        (job_id, uid, target_handle, now, now),
    )
    conn.commit()
    return job_id


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: str | None = None,
    result: str | None = None,
    error: str | None = None,
):
    """Update job fields. Only non-None fields are written."""
    conn = get_connection()
    fields = []
    values = []

    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    if result is not None:
        fields.append("result = ?")
        values.append(result)
    if error is not None:
        fields.append("error = ?")
        values.append(error)

    if not fields:
        return

    fields.append("updated_at = ?")
    values.append(_utcnow_iso())
    values.append(job_id)

    conn.execute(
        f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?",
        values,
    )
    conn.commit()


def get_job(job_id: str) -> dict | None:
    """Fetch a job row as a dict. Returns None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()

    if row is None:
        return None

    return {
        "job_id": row["job_id"],
        "uid": row["uid"],
        "target_handle": row["target_handle"],
        "status": row["status"],
        "progress": json.loads(row["progress"]) if row["progress"] else None,
        "result": json.loads(row["result"]) if row["result"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ── Helpers ───────────────────────────────────────────────────────

def _utcnow_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()

/**
 * API client for the MGTAB Bot Detector backend.
 * 
 * Queue-driven architecture:
 * - submitPrediction()  → POST /api/v1/predict (returns job_id)
 * - getJobStatus()      → GET /api/v1/jobs/{job_id} (poll for results)
 * - predictUser()       → POST /predict/user (manual mode, synchronous)
 *
 * All authenticated requests include Authorization: Bearer <JWT>.
 */

import { auth } from '../firebase';

// For local testing:
// const API_BASE = 'http://localhost:8000';
// For production:
const API_BASE = 'https://arihant0008-mgtab-bot-detector-main.hf.space';


// ── Auth Helper ──────────────────────────────────────────────────

/**
 * Get the current user's Firebase ID token for HTTP requests.
 * Returns headers object with Authorization: Bearer <token>.
 * Falls back to empty object if no user is signed in.
 */
async function getAuthHeaders() {
  try {
    if (auth.currentUser) {
      const token = await auth.currentUser.getIdToken();
      return { 'Authorization': `Bearer ${token}` };
    }
  } catch (err) {
    console.warn('Failed to get auth token:', err);
  }
  return {};
}


// ── Queue-Driven Endpoints ───────────────────────────────────────

/**
 * Submit a bot detection job for a Twitter/X handle.
 * Returns { job_id, status, poll_url, from_cache? }.
 * 
 * @throws {Object} { code: 'RATE_LIMITED', retryAfterSeconds, message } on 429
 * @throws {Error} on other failures
 */
export async function submitPrediction(handle, refresh = false) {
  const authHeaders = await getAuthHeaders();
  const cleanHandle = handle.replace(/^@/, '').trim();

  const response = await fetch(`${API_BASE}/api/v1/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders,
    },
    body: JSON.stringify({
      target_handle: cleanHandle,
      refresh,
    }),
  });

  if (response.status === 429) {
    const err = await response.json().catch(() => ({}));
    const detail = err.detail || {};
    throw {
      code: 'RATE_LIMITED',
      retryAfterSeconds: detail.retry_after_seconds || 86400,
      message: detail.message || 'Rate limit exceeded. Try again later.',
    };
  }

  if (response.status === 422) {
    const err = await response.json().catch(() => ({}));
    throw new Error('Invalid handle format. Use only letters, numbers, and underscores (max 15 chars).');
  }

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Submission failed (${response.status})`);
  }

  return response.json();
}


/**
 * Poll the status of a submitted job.
 * Returns { job_id, status, progress, result, error, created_at }.
 */
export async function getJobStatus(jobId) {
  const authHeaders = await getAuthHeaders();

  const response = await fetch(`${API_BASE}/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    headers: { ...authHeaders },
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch job status (${response.status})`);
  }

  return response.json();
}


// ── Manual Mode (POST /predict/user) ─────────────────────────────

export async function predictUser(requestData) {
  const authHeaders = await getAuthHeaders();

  const response = await fetch(`${API_BASE}/predict/user`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders,
    },
    body: JSON.stringify(requestData),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Prediction failed (${response.status})`);
  }

  return response.json();
}


// ── Utility endpoints ────────────────────────────────────────────

export async function getModelInfo() {
  const authHeaders = await getAuthHeaders();
  const response = await fetch(`${API_BASE}/model/info`, {
    headers: { ...authHeaders },
  });
  if (!response.ok) throw new Error('Failed to fetch model info');
  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Backend unreachable');
  return response.json();
}

export async function getFeaturesSchema() {
  const authHeaders = await getAuthHeaders();
  const response = await fetch(`${API_BASE}/features/schema`, {
    headers: { ...authHeaders },
  });
  if (!response.ok) throw new Error('Failed to fetch features schema');
  return response.json();
}

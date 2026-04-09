/**
 * API client for the MGTAB Bot Detector backend.
 */

// For local testing:
// const API_BASE = 'http://localhost:8000';
// For production, switch back to:
const API_BASE = 'https://arihant0008-mgtab-detector-api.hf.space';

export async function predictUser(requestData) {
  const response = await fetch(`${API_BASE}/predict/user`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestData),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Prediction failed (${response.status})`);
  }

  return response.json();
}

export async function getModelInfo() {
  const response = await fetch(`${API_BASE}/model/info`);
  if (!response.ok) throw new Error('Failed to fetch model info');
  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Backend unreachable');
  return response.json();
}

export async function getFeaturesSchema() {
  const response = await fetch(`${API_BASE}/features/schema`);
  if (!response.ok) throw new Error('Failed to fetch features schema');
  return response.json();
}

/**
 * API client for the MGTAB Bot Detector backend.
 * Supports both manual prediction (POST) and one-click SSE (EventSource).
 */

// For local testing:
const API_BASE = 'http://localhost:8000';
// For production:
// const API_BASE = 'https://arihant0008-mgtab-bot-detector-main.hf.space';


// ── Manual Mode (POST /predict/user) ─────────────────────────────

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


// ── One-Click SSE Mode (GET /predict/username/{handle}) ──────────

/**
 * Connect to the SSE endpoint for one-click username analysis.
 *
 * @param {string} username  — Twitter handle (with or without @)
 * @param {object} callbacks — Event handlers:
 *   - onProgress({step, status, message})
 *   - onScrapComplete(scrapeMeta)
 *   - onResult(predictionResult)
 *   - onError(errorObj)
 *   - onDone()
 * @returns {function} abort — Call this to cancel the stream
 */
export function predictByUsername(username, callbacks = {}) {
  const handle = username.replace(/^@/, '').trim();
  if (!handle) {
    callbacks.onError?.({ message: 'Username cannot be empty', status_code: 400 });
    return () => {};
  }

  const controller = new AbortController();

  // Use fetch + ReadableStream for SSE (more control than EventSource for error handling)
  (async () => {
    try {
      const response = await fetch(`${API_BASE}/predict/username/${encodeURIComponent(handle)}`, {
        signal: controller.signal,
        headers: { 'Accept': 'text/event-stream' },
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        callbacks.onError?.({
          message: err.detail || `Request failed (${response.status})`,
          status_code: response.status,
        });
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE events from the buffer
        const events = buffer.split('\n\n');
        buffer = events.pop(); // Keep incomplete event in buffer

        for (const eventBlock of events) {
          if (!eventBlock.trim()) continue;

          let eventType = 'message';
          let eventData = '';

          for (const line of eventBlock.split('\n')) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6);
            } else if (line.startsWith(':')) {
              // Comment/keepalive — ignore
              continue;
            }
          }

          if (!eventData) continue;

          try {
            const data = JSON.parse(eventData);

            switch (eventType) {
              case 'progress':
                callbacks.onProgress?.(data);
                break;
              case 'scrape_complete':
                callbacks.onScrapeComplete?.(data);
                break;
              case 'cache_hit':   // Redis cache hit — same shape as result
              case 'result':
                callbacks.onResult?.(data);
                break;
              case 'error':
                callbacks.onError?.(data);
                break;
              case 'done':
                callbacks.onDone?.();
                break;
            }
          } catch (parseErr) {
            console.warn('SSE parse error:', parseErr, eventData);
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return; // User cancelled
      callbacks.onError?.({
        message: err.message || 'Connection failed',
        status_code: 0,
      });
    }
  })();

  // Return abort function  
  return () => controller.abort();
}


// ── Utility endpoints ────────────────────────────────────────────

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

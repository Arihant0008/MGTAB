
<center><b>CHAPTER 3 <br> SYSTEM ARCHITECTURE AND DESIGN</b></center>

&nbsp;

## **3.1 HIGH-LEVEL ARCHITECTURE**

The MGTAB Bot Detector is implemented as a client-server web application with a clear separation between the presentation layer, the application logic, and the machine learning inference pipeline. The system comprises three principal components:

1. **Frontend (React 19 + Vite)**: A single-page application served as static assets through Vercel CDN. The frontend handles user input, renders the real-time progress stepper, and displays prediction results. It communicates with the backend via HTTP REST calls and Server-Sent Events (SSE).

2. **Backend (FastAPI + Python 3.11)**: An asynchronous Python server responsible for Twitter data scraping, feature engineering, graph construction, and RGCN inference. The backend exposes five API endpoints and manages the full detection pipeline from username input to classification output.

3. **External Services**: The Twitter/X platform (accessed via Scweet's internal GraphQL API), the Hugging Face model hub (for LaBSE weights), and the deployment platforms (Hugging Face Spaces for the backend Docker container, Vercel for the frontend).

[Insert Figure 3.1: High-Level System Architecture Here]

The architectural decision to separate the frontend and backend into independently deployable units was driven by two considerations. First, the backend's machine learning dependencies (PyTorch, PyTorch Geometric, Transformers) result in a Docker image exceeding 1.5 GB, which is ill-suited for hosting alongside static frontend assets. Second, independent deployment enables the frontend to be served from a global CDN (Vercel) with sub-50ms latency to end users, while the backend can be hosted on GPU-capable infrastructure (Hugging Face Spaces) optimised for inference workloads.

## **3.2 FRONTEND ARCHITECTURE**

### **3.2.1 React Component Hierarchy**

The frontend is built with React 19 using functional components and hooks. Vite 8 serves as the build tool and development server, providing hot module replacement (HMR) and optimised production bundling. The application uses React Router v7 for client-side routing across three pages.

The component hierarchy is structured as follows:

```
App.jsx (Root — BrowserRouter)
├── Navbar.jsx (Global navigation bar)
├── HomePage.jsx (Landing page — route: /)
│   ├── Hero.jsx (Hero section with tagline and CTA)
│   └── ModelStats.jsx (Live model performance metrics)
├── DetectorPage.jsx (Main detection UI — route: /detect)
│   ├── ProfileForm.jsx (20-field profile data input form)
│   ├── TweetInput.jsx (Dynamic tweet text list editor)
│   ├── RelationsEditor.jsx (Relation/edge editor with type selector)
│   └── ResultCard.jsx (Prediction result display)
└── AnalyticsPage.jsx (Model statistics — route: /analytics)
```

The `DetectorPage` component is the most complex, managing two distinct operational modes: **Auto Mode** (one-click SSE-based analysis) and **Manual Mode** (direct data entry with POST request). This dual-mode design provides resilience against Twitter API outages — when automated scraping is unavailable due to rate limiting or authentication token expiry, users can still access the RGCN classifier by entering profile data manually.

[Insert Figure 3.2: React Component Hierarchy Diagram Here]

**Table 3.2: Frontend Route Definitions**

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `HomePage` | Landing page with hero section and model statistics |
| `/detect` | `DetectorPage` | Primary detection interface (auto + manual modes) |
| `/analytics` | `AnalyticsPage` | Detailed model performance metrics and research context |

### **3.2.2 Client-Side State Management**

State management in the frontend uses React's built-in `useState` and `useCallback` hooks, without external state management libraries (e.g., Redux, Zustand). This decision was motivated by the observation that the application's state is localised almost entirely within the `DetectorPage` component and does not require global state sharing across routes.

The `DetectorPage` maintains the following state variables:

- **Mode state**: `mode` — toggles between `'auto'` and `'manual'` modes.
- **Auto mode state**: `username`, `currentStep` (integer 0–6), `stepMessage`, `scrapeMeta`, `autoResult`, `autoLoading`, `autoError`.
- **Manual mode state**: `profile` (object with 20 fields), `tweets` (string array), `relations` (array of edge objects), `manualResult`, `manualLoading`, `manualError`.
- **Abort reference**: `abortRef` — a `useRef` holding the abort function returned by the SSE client, enabling cancellation of in-flight analysis requests.

The `useCallback` hook is used for event handlers (`handleAutoSubmit`, `handleAutoReset`) to prevent unnecessary re-renders of child components.

### **3.2.3 SSE Stream Consumer**

The frontend's SSE consumer is implemented in `predict.js` using the Fetch API with `ReadableStream`, rather than the browser's native `EventSource` API. This design choice provides finer control over error handling and connection lifecycle management. The implementation proceeds as follows:

1. A `fetch()` request is issued to `GET /predict/username/{handle}` with the `Accept: text/event-stream` header.
2. The response body is read incrementally using `response.body.getReader()`.
3. Incoming byte chunks are decoded via `TextDecoder` and accumulated in a buffer.
4. The buffer is split on double-newline (`\n\n`) boundaries to extract complete SSE events.
5. Each event is parsed by extracting the `event:` type and `data:` payload lines.
6. Events are dispatched to the appropriate callback: `onProgress`, `onScrapeComplete`, `onResult`, `onError`, or `onDone`.

An `AbortController` is attached to the fetch request, and its `abort()` function is returned to the caller. This enables the user to cancel an in-progress analysis — for instance, if they realise they entered the wrong username.

[Insert Figure 3.3: SSE Data Flow Sequence Diagram Here]

## **3.3 BACKEND ARCHITECTURE**

### **3.3.1 FastAPI Application Structure**

The backend is implemented as a FastAPI application with the following module structure:

```
backend/app/
├── __init__.py            # Package marker
├── main.py                # FastAPI app, endpoints, lifespan
├── scraper.py             # Scweet-based Twitter scraper (809 lines)
├── features.py            # Feature engineering pipeline (220 lines)
├── graph_builder.py       # PyG graph construction (220 lines)
├── inference.py           # RGCN model loading and prediction (105 lines)
├── rgcn_model.py          # RGCN architecture definition (39 lines)
├── normalization.py       # MinMax normalization constants (98 lines)
└── config.py              # Environment and model constants (80 lines)
```

The application uses FastAPI's `lifespan` context manager for startup and shutdown logic. At startup, the `InferenceEngine` singleton is instantiated, which loads the trained RGCN weights (`best_rgcn.pt`, 6.5 MB) from disk and places the model in evaluation mode. The Scweet scraper singleton is also initialised at startup (with deferred authentication — the actual Twitter login occurs on the first scraping request).

CORS middleware is configured to accept requests from all origins (`allow_origins=["*"]`), which is necessary because the frontend and backend are served from different domains in production.

### **3.3.2 API Route Design**

The backend exposes five endpoints:

**Table 3.1: FastAPI Endpoint Summary**

| Method | Path | Purpose | Response Type |
|--------|------|---------|--------------|
| `POST` | `/predict/user` | Manual prediction with raw JSON body | JSON (`PredictResponse`) |
| `GET` | `/predict/username/{handle}` | One-click SSE analysis via scraping | `text/event-stream` |
| `GET` | `/model/info` | Model metadata (architecture, accuracy) | JSON |
| `GET` | `/health` | Backend health check | JSON |
| `GET` | `/features/schema` | Feature definitions for frontend form | JSON |

The `POST /predict/user` endpoint accepts a `PredictRequest` body validated by Pydantic v2 models. The request body contains three fields:
- `target`: an object with `profile` (20 fields) and `tweets` (string array).
- `neighbors`: an array of neighbor objects, each with `id`, `profile`, and `tweets`.
- `relations`: an array of edge objects with `source`, `target`, and `relation` fields.

The `GET /predict/username/{handle}` endpoint initiates the full automated pipeline: scraping, feature engineering, graph construction, and RGCN inference. It returns a `StreamingResponse` with media type `text/event-stream`, emitting SSE events as the pipeline progresses.

[Insert Figure 3.4: FastAPI Route Structure Here]

### **3.3.3 Asynchronous Task Handling and SSE**

The SSE endpoint employs an `asyncio.Queue`-based pattern to bridge the asynchronous scraping task with the SSE event stream generator. The implementation works as follows:

1. A `progress_queue` (of type `asyncio.Queue`) is created.
2. A `progress_callback` coroutine is defined that pushes `(event_type, event_data)` tuples to the queue when called by the scraper.
3. The scraping coroutine (`scraper.scrape_ego_graph()`) is wrapped in `asyncio.create_task()`, which starts it running concurrently.
4. The SSE generator function enters a loop: at each iteration, it attempts to dequeue a progress event with a 0.5-second timeout. If a progress event is available, it is yielded as an SSE event. If the timeout fires (no new progress), a keepalive comment (`: keepalive\n\n`) is yielded to prevent the HTTP connection from being closed by intermediary proxies.
5. When the scraping task completes, remaining events are drained from the queue, and the RGCN inference step is executed synchronously (inference takes less than one second). The final `result` event is emitted, followed by a `done` event.

This pattern avoids blocking the FastAPI event loop during the long-running scraping process (typically 60–90 seconds) while providing real-time progress feedback to the frontend. The keepalive comments are particularly important for cloud deployments (Vercel, Hugging Face Spaces) that impose idle connection timeouts of 30–60 seconds.

## **3.4 DEPLOYMENT INFRASTRUCTURE**

The system is deployed using a two-platform architecture:

**Backend — Hugging Face Spaces (Docker SDK)**: The backend is containerised using a `Dockerfile` based on `python:3.11-slim`. The Docker build process installs PyTorch (CPU-only variant, reducing image size from ~3 GB to ~1.5 GB), PyTorch Geometric, the Transformers library, and FastAPI. Hugging Face Spaces automatically routes HTTPS traffic to port 7860 inside the container. The `TWITTER_AUTH_TOKEN` is stored as a Hugging Face Repository Secret, which is injected as an environment variable at runtime.

**Frontend — Vercel**: The React application is built into static assets via `vite build` and deployed through Vercel's Git integration. Vercel provides automatic HTTPS, global CDN distribution, and continuous deployment on each push to the main branch.

[Insert Figure 3.5: Deployment Architecture (Vercel + Hugging Face Spaces) Here]

The live URL https://www.mgtab.me/ points to the Vercel-hosted frontend, which communicates with the Hugging Face Spaces-hosted backend at its `.hf.space` URL.

&nbsp;

---

<center><b>CHAPTER 4 <br> DATA INGESTION AND FEATURE ENGINEERING</b></center>

&nbsp;

## **4.1 TWITTER DATA ACQUISITION PIPELINE**

### **4.1.1 Cookie-Based Authentication**

The Twitter API v2 requires a paid developer account (minimum $100/month) for programmatic access to user profiles, tweets, and social graph data. This project bypasses the official API entirely by using Scweet v5, an open-source Python library that accesses Twitter's internal GraphQL API — the same API used by the web client at x.com.

Authentication is performed using a browser cookie (`auth_token`) extracted from an active Twitter session. The process is as follows:

1. The developer logs into Twitter/X via a web browser.
2. Using the browser's Developer Tools (F12 → Application → Cookies → x.com), the value of the `auth_token` cookie is copied.
3. This value is set as the `TWITTER_AUTH_TOKEN` environment variable in the backend's `.env` file (or as a Hugging Face Repository Secret for production).
4. At initialisation, the Scweet library uses this cookie to authenticate all subsequent GraphQL requests.

This approach has the advantage of requiring zero API keys and incurring zero monetary cost. The `auth_token` cookie typically remains valid for 12–24 months. However, it has the inherent limitation that it relies on Twitter's internal API, which is undocumented and subject to change without notice.

**Table 4.1: Scweet Scraper Configuration Parameters**

| Parameter | Value | Source |
|-----------|-------|--------|
| `auth_token` | Browser cookie value | `TWITTER_AUTH_TOKEN` env var |
| `daily_requests_limit` | 500 | ScweetConfig default |
| `daily_tweets_limit` | 5000 | ScweetConfig default |
| `min_delay_s` | 1.0 s | ScweetConfig default |
| `SCRAPE_DELAY_SECONDS` | 3.0 s | Environment variable |
| `MAX_NEIGHBORS_PER_RELATION` | 10 | `config.py` constant |
| `MAX_TWEETS_TARGET` | 20 | `config.py` constant |
| `MAX_TWEETS_NEIGHBOR` | 5 | `config.py` constant |
| Proxy | Optional | `PROXY_URL` env var |

### **4.1.2 Ego-Graph Scraping Strategy**

For each detection query, the system constructs an ego-graph centred on the target user. The scraping pipeline executes in four sequential steps:

**Step 1 — Scrape Target Profile and Tweets**: The target user's profile metadata is fetched via `aget_user_info([username])`, which returns a dictionary containing follower count, friend count, listed count, statuses count, favourites count, name, screen name, description, creation date, verification status, and profile image URL. Subsequently, up to 20 recent tweets are fetched via `aget_profile_tweets([username], limit=20)`. Tweet texts, URLs, and hashtags are extracted from the raw tweet objects for use in feature engineering and co-occurrence edge discovery.

**Step 2 — Discover Neighbours**: The target's neighbourhood is explored along five relation types:
- **Followers**: Up to 10 users who follow the target (`aget_followers()`).
- **Friends**: Up to 10 users the target follows (`aget_following()`).
- **Mentions**: Usernames extracted from `@mention` patterns in the target's tweets.
- **Replies**: Usernames extracted from the `in_reply_to_screen_name` field in the tweet's raw GraphQL payload.
- **Quotes**: Usernames extracted from `twitter.com/user/status/` URL patterns in quoted tweets.

All discovered neighbours are deduplicated by screen name, producing a set of approximately 20–50 unique neighbour candidates.

**Step 3 — Enrich Neighbours**: For each unique neighbour, the system fetches their profile metadata and up to 5 recent tweets. This step is the most time-consuming, requiring up to 40 individual API calls (2 per neighbour × 20 neighbours), with a configurable delay of 3.0 seconds between calls. URLs and hashtags from each neighbour's tweets are tracked for opportunistic co-occurrence edge discovery.

**Step 4 — Build Relations**: The collected data is assembled into a list of relation edges. For each neighbour, edges are created for each relation type through which they were discovered. Additionally, URL co-occurrence edges are added where the target and a neighbour share at least one common URL, and hashtag co-occurrence edges are added where they share at least one common hashtag.

[Insert Figure 4.1: Ego-Graph Scraping Pipeline Flowchart Here]

### **4.1.3 Rate-Limit Resilience**

Twitter's internal API enforces rate limits, returning HTTP 429 responses when request quotas are exceeded. The scraping pipeline implements a graceful degradation strategy:

- All tweet-fetching calls (for both the target and neighbours) are wrapped in try-except blocks that catch `RateLimitError` and HTTP 429 patterns.
- If a neighbour's tweets cannot be fetched due to rate limiting, the neighbour still enters the graph with profile features only; its 768-dimensional tweet embedding is set to a zero vector.
- A configurable delay (`SCRAPE_DELAY_SECONDS`, default 3.0 seconds) is inserted between consecutive API calls to reduce the likelihood of triggering rate limits.
- The Scweet library's daily request and tweet counters are reset at each server startup to prevent false lockouts from previous sessions.

[Insert Figure 4.2: Rate-Limit Fallback Decision Flow Here]

This design decision ensures that the detection pipeline produces a result even under adverse scraping conditions. In the worst case — where all tweet fetches are rate-limited — the graph degrades to a profile-only classification with zero-vector tweet embeddings, which is less accurate but still functional.

## **4.2 FEATURE VECTOR CONSTRUCTION**

### **4.2.1 Profile Feature Extraction (20-D)**

Each node in the graph is represented by a 788-dimensional feature vector. The first 20 dimensions encode profile metadata, structured as follows:

**Table 4.2: Profile Feature Definitions and Index Mapping**

| Index | Feature Name | Type | Description |
|-------|-------------|------|-------------|
| 0 | `profile_use_background_image` | Boolean | Whether the profile uses a custom background image |
| 1 | `default_profile` | Boolean | Whether the profile theme is unchanged from default |
| 2 | `verified` | Boolean | Official verification status (or blue verified) |
| 3 | `followers_count` | Numerical | Number of followers (log + MinMax scaled) |
| 4 | `default_profile_image` | Boolean | Whether the profile uses the default avatar |
| 5 | `listed_count` | Numerical | Number of public lists the user appears in |
| 6 | `statuses_count` | Numerical | Total number of tweets and retweets |
| 7 | `friends_count` | Numerical | Number of accounts the user follows |
| 8 | `geo_enabled` | Boolean | Whether geolocation is enabled |
| 9 | `favourites_count` | Numerical | Total number of tweets liked |
| 10 | `created_at` | Numerical | Account creation timestamp (log-nanoseconds + MinMax) |
| 11 | `screen_name_length` | Derived | Length of the user's handle (MinMax scaled) |
| 12 | `name_length` | Derived | Length of the display name (MinMax scaled) |
| 13 | `description_length` | Derived | Length of the bio description (MinMax scaled) |
| 14 | `followers_friends_ratios` | Derived | Ratio of followers to friends (log + MinMax) |
| 15 | `default_profile_background_color` | Boolean | Whether the background colour is the default |
| 16 | `default_profile_sidebar_fill_color` | Boolean | Whether the sidebar fill is the default |
| 17 | `default_profile_sidebar_border_color` | Boolean | Whether the sidebar border is the default |
| 18 | `has_URL` | Boolean | Whether the profile contains a URL |
| 19 | `profile_background_image_URL` | Boolean | Whether a background image URL is set |

[Insert Figure 4.3: 788-Dimensional Feature Vector Layout Here]

Boolean features are encoded directly as 0.0 or 1.0. Numerical and derived features undergo the normalisation pipeline described in Section 4.2.2.

Six of the 20 features (indices 8, 15, 16, 17, 0, and 19) correspond to legacy Twitter API fields that are no longer exposed by the modern internal API. The `scweet_user_to_profile()` adapter function injects dataset-modal default values for these fields: `geo_enabled = False`, `profile_use_background_image = True`, `default_profile_background_color = False`, `default_profile_sidebar_fill_color = False`, `default_profile_sidebar_border_color = False`, and `profile_background_image_url = False`. These defaults were determined by computing the statistical mode of each field across the MGTAB training dataset.

### **4.2.2 Normalization Pipeline**

Numerical features are normalised to the [0, 1] range using a two-step pipeline that matches the preprocessing used during MGTAB dataset construction:

**Step 1 — Log Transform**: For count-based features (followers, friends, listed, statuses, favourites, and the followers/friends ratio), a log(1 + x) transformation is applied to compress the dynamic range. Twitter counts span several orders of magnitude (e.g., followers ranging from 0 to hundreds of millions), and the log transform prevents high-count accounts from dominating the feature space.

**Step 2 — MinMax Scaling**: The log-transformed value is scaled to [0, 1] using bounds derived from the MGTAB training dataset:

scaled(x) = (x - x_min) / (x_max - x_min), clamped to [0, 1]

**Table 4.3: MinMax Normalization Bounds from MGTAB Dataset**

| Feature | Min (log-transformed) | Max (log-transformed) |
|---------|-----------------------|-----------------------|
| `followers_count` | 0.0 | 25.573 |
| `friends_count` | 0.0 | 21.030 |
| `listed_count` | 0.0 | 17.675 |
| `statuses_count` | 0.0 | 20.386 |
| `favourites_count` | 0.0 | 19.711 |
| `created_at` | 36.554 | 51.711 |
| `screen_name_length` | 3.0 | 15.0 |
| `name_length` | 1.0 | 50.0 |
| `description_length` | 0.0 | 204.0 |
| `followers_friends_ratios` | 0.0 | 11.169 |

The `created_at` feature receives special treatment. The MGTAB dataset encodes account creation time as log(timestamp in nanoseconds), producing values in the range [36.55, 51.71]. This encoding was reverse-engineered during the project: log(unix_seconds) yields values around 21, which falls below the dataset minimum. Multiplying by 10^9 (converting to nanoseconds) before taking the logarithm produces values (~41.6–42.0 for accounts created between 2008 and 2024) that correctly fall within the dataset bounds.

[Insert Figure 4.4: Log-MinMax Normalization Pipeline Here]

### **4.2.3 LaBSE Tweet Embedding (768-D)**

The remaining 768 dimensions of the feature vector are produced by the Language-Agnostic BERT Sentence Embedding (LaBSE) model, a multilingual sentence encoder pre-trained by Google that maps text strings to 768-dimensional dense vectors.

The tweet embedding pipeline is as follows:

1. All valid tweet texts for a user are collected (up to 20 for the target, up to 5 for each neighbour).
2. The texts are tokenised using the LaBSE tokenizer with padding, truncation at 128 tokens, and conversion to PyTorch tensors.
3. The tokenised batch is passed through the LaBSE model in no-gradient mode.
4. The `pooler_output` (the [CLS] token embedding) is extracted for each tweet, producing a tensor of shape (num_tweets, 768).
5. The per-tweet embeddings are **summed** (not averaged) across the tweet dimension, producing a single 768-dimensional vector.

The choice of summation over averaging is significant. During the development of this project, it was discovered that the MGTAB training data was constructed using summed raw `pooler_output` values — not L2-normalised, and not averaged. The resulting embedding norms in the training data are approximately 18–20. When L2 normalisation was mistakenly applied during inference (shrinking norms to ~0.5), the trained RGCN weights assigned negligible importance to the 768 tweet dimensions relative to the 20 profile dimensions, causing dramatic classification errors. Switching from normalised-averaged embeddings to raw-summed embeddings resolved this issue and produced predictions consistent with the training data distribution.

[Insert Figure 4.5: LaBSE Encoding and Summation Pipeline Here]

If no tweets are available for a user (either because the account has no tweets or because tweet fetching was rate-limited), a zero vector of dimension 768 is used instead.

### **4.2.4 Feature Concatenation**

The final node feature vector is obtained by concatenating the 20-dimensional profile vector and the 768-dimensional tweet vector:

**f** = [ profile_features || tweet_embedding ] ∈ ℝ^788

The `build_node_feature()` function in `features.py` performs this concatenation for each node in the graph.

## **4.3 MULTI-RELATIONAL GRAPH CONSTRUCTION**

### **4.3.1 Relation Type Definitions**

The MGTAB framework defines seven relation types, each capturing a distinct form of interaction or association between Twitter accounts:

**Table 4.4: Seven MGTAB Relation Types with Direction Semantics**

| ID | Relation | Type | Direction | Discovery Method |
|----|----------|------|-----------|-----------------|
| 0 | Follower | Explicit | neighbour → target | `aget_followers()` API |
| 1 | Friend | Explicit | target → neighbour | `aget_following()` API |
| 2 | Mention | Explicit | target → neighbour | `@username` regex in tweets |
| 3 | Reply | Explicit | target → neighbour | `in_reply_to_screen_name` in GraphQL |
| 4 | Quoted | Explicit | target → neighbour | Tweet URL pattern matching |
| 5 | URL | Implicit | bidirectional ↔ | URL co-occurrence in tweets |
| 6 | Hashtag | Implicit | bidirectional ↔ | Hashtag co-occurrence in tweets |

[Insert Figure 4.6: Seven Relation Types in the MGTAB Graph Here]

### **4.3.2 Edge Direction Semantics**

Edge directions follow the conventions established in the MGTAB paper (Table 4, Shi et al., 2023):

- **Follower** (type 0): The edge flows *from* the neighbour *to* the target, reflecting the fact that the neighbour is a follower of the target.
- **Friend** (type 1): The edge flows *from* the target *to* the neighbour, reflecting the fact that the target follows the neighbour.
- **Mention, Reply, Quoted** (types 2, 3, 4): These edges flow *from* the target *to* the neighbour, since the target is the actor (the one mentioning, replying, or quoting).
- **URL and Hashtag** (types 5, 6): These are undirected relations, implemented by adding edges in both directions (target → neighbour and neighbour → target) with the same relation type.

The `REVERSE_SOURCE_RELATIONS` set in `config.py` identifies relation type 0 (follower) as requiring direction reversal, and the `UNDIRECTED_RELATIONS` set identifies types 5 and 6 as requiring bidirectional edge insertion.

### **4.3.3 Graph Filtering and Self-Loop Fallback**

The `build_mini_graph()` function in `graph_builder.py` applies two important filtering steps before assembling the PyTorch Geometric `Data` object:

**Neighbour filtering**: Only neighbours with "real data" — defined as having at least one non-empty tweet text or at least one non-zero/non-empty meaningful profile field — are included in the graph. This filtering is critical because zero-vector placeholder nodes corrupt the RGCN's mean aggregation function. The trained model learned its weights with all nodes having valid 788-dimensional features; injecting nodes with all-zero features introduces out-of-distribution inputs that distort the neighbourhood aggregation and produce unreliable predictions.

**Self-loop fallback**: If no valid edges remain after filtering (e.g., because no neighbours had real data), a self-loop edge of type 0 is added from the target node to itself. This ensures that the RGCN can still execute a forward pass, operating in "feature-only mode" where the prediction is based solely on the target node's own feature vector (transformed through two RGCN layers with self-connection weights).

The function returns a PyTorch Geometric `Data` object containing:
- `x`: a float32 tensor of shape (num_nodes, 788) — the node feature matrix.
- `edge_index`: a long tensor of shape (2, num_edges) — the source and destination indices for each edge.
- `edge_type`: a long tensor of shape (num_edges,) — the relation type index (0–6) for each edge.

[Insert Figure 4.7: Example Mini Ego-Graph for Inference Here]

&nbsp;

---

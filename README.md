<div align="center">

# 🤖 MGTAB Bot Detector

### Multi-Relational Graph-Based Twitter/X Account Bot Detection

[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Redis](https://img.shields.io/badge/Redis-Upstash-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://upstash.com)
[![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-Space-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Arihant0008/mgtab-bot-detector-main)

**Detect sophisticated Twitter/X bots using graph neural networks — not just metadata.**

[Live Demo](https://www.mgtab.me/) · [Paper Reference](https://arxiv.org/abs/2301.12174) · [Report Bug](https://github.com/Arihant0008/MGTAB/issues)

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [System Architecture](#-system-architecture)
- [Data Flow Pipeline](#-data-flow-pipeline)
- [Key Technical Implementations](#-key-technical-implementations)
- [Redis Caching Layer](#-redis-caching-layer)
- [Confidence Calibration](#-confidence-calibration)
- [Frontend Architecture](#-frontend-architecture)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Setup & Installation](#-setup--installation)
- [Deployment](#-deployment)
- [Limitations & Future Work](#-limitations--future-work)
- [References](#-references)

---

## 🎯 Problem Statement

Modern social media bots have evolved far beyond simple scripted accounts. Today's sophisticated bots:

- **Forge realistic profiles** with AI-generated bios, profile pictures, and follower graphs
- **Leverage LLMs** (GPT, Claude) to produce human-like tweet content
- **Mimic organic behavior** with variable posting schedules and engagement patterns

Traditional bot detection methods relying solely on **metadata analysis** (follower count, account age, tweet frequency) fail against these next-generation adversaries. The key insight: **bots can fake profiles, but they cannot easily forge authentic social graph topology.**

---

## 💡 Solution Overview

This project implements the **MGTAB benchmark** (Shi et al., 2023) as a production-ready, full-stack web application. Instead of analyzing accounts in isolation, we construct a **multi-relational ego-graph** around the target user and leverage graph neural networks to detect structural anomalies invisible to traditional classifiers.

### How It Works

| Component | Description |
|---|---|
| **Feature Vector** | 788-dimensional: 20 normalized profile features + 768-D LaBSE tweet embeddings |
| **Graph Structure** | 7-edge multi-relational ego-graph (follower, friend, mention, reply, quote, URL, hashtag) |
| **Model** | 2-layer Relational Graph Convolutional Network (RGCN) via PyTorch Geometric |
| **Inference** | Real-time scrape → encode → classify pipeline with Server-Sent Events streaming |

---

## 🏗 System Architecture

```mermaid
graph TB
    subgraph Frontend ["🖥️ Frontend — React 19 + Vite + React Router"]
        UI["Search Bar<br/>@username Input"]
        Stepper["5-Step SSE<br/>Progress Stepper"]
        Result["Prediction Card<br/>Human / Bot"]
        Analytics["Analytics Page<br/>Model Comparisons"]
    end

    subgraph Backend ["⚙️ Backend — FastAPI + Python 3.11"]
        API["FastAPI Router<br/>SSE Endpoint"]
        Cache["Redis Cache<br/>(Upstash REST)"]
        
        subgraph Scraper ["🔍 Scweet v5.3 Scraper"]
            Auth["Cookie Auth<br/>(TWITTER_AUTH_TOKEN)"]
            Profile["Scrape Profile<br/>+ 20 Tweets"]
            Ego["Ego-Graph<br/>Discovery"]
            Enrich["Neighbor<br/>Enrichment"]
        end
        
        subgraph ML ["🧠 ML Pipeline"]
            Adapter["Profile Adapter<br/>20 MGTAB Features"]
            LaBSE["LaBSE Encoder<br/>768-D Embeddings"]
            GraphB["PyG Graph<br/>Builder"]
            RGCN["RGCN Model<br/>788→256→2"]
            Calib["Confidence<br/>Calibration"]
        end
    end

    subgraph External ["🌐 External Services"]
        Twitter["Twitter/X<br/>GraphQL API"]
        Redis["Upstash Redis<br/>(HTTPS REST)"]
        HF["🤗 Hugging Face<br/>Spaces (Docker)"]
        Vercel["▲ Vercel<br/>CDN"]
    end

    UI -->|"GET /predict/username/@handle"| API
    API -->|"Check Cache"| Cache
    Cache -.->|"HIT → instant"| Result
    Cache -.->|"MISS"| Auth
    API --> Auth
    Auth -->|"Cookie + Proxy"| Twitter
    Auth --> Profile
    Profile --> Ego
    Ego --> Enrich
    Enrich --> Adapter
    Adapter --> LaBSE
    LaBSE --> GraphB
    GraphB --> RGCN
    RGCN --> Calib
    Calib -->|"SSE Stream"| Stepper
    Stepper --> Result
    RGCN -.->|"Cache SET"| Cache
    Cache -.->|"REST API"| Redis
    
    Frontend -.->|"Deployed on"| Vercel
    Backend -.->|"Deployed on"| HF
```

---

## 🔄 Data Flow Pipeline

```mermaid
flowchart LR
    A["👤 @username"] --> B["🔑 Authenticate<br/>(Cookie Auth)"]
    B --> C["📊 Scrape Target<br/>Profile + 20 Tweets"]
    C --> D["🕸️ Discover Ego-Graph<br/>5 Relations × 10 Neighbors"]
    D --> E["📥 Enrich Neighbors<br/>Profile + 5 Tweets Each"]
    E --> F["🔢 Encode Features<br/>20 Profile + 768 LaBSE"]
    F --> G["📐 Build PyG Graph<br/>7 Edge Types"]
    G --> H["🧠 RGCN Forward Pass<br/>788→256→2"]
    H --> I["📊 Softmax<br/>P(human) + P(bot)"]
    I --> J["✅ Result<br/>Human or Bot"]
    
    style A fill:#1DA1F2,color:#fff
    style J fill:#22c55e,color:#fff
    style H fill:#EE4C2C,color:#fff
```

### Pipeline Timing (Typical)

| Stage | Duration | API Calls |
|---|---|---|
| Authentication | ~3s | 1 (bootstrap) |
| Target Scrape | ~4s | 2 (profile + tweets) |
| Ego-Graph Discovery | ~12s | 2 (followers + following) |
| Neighbor Enrichment | ~60s | Up to 40 (profile + tweets × 20) |
| Feature Encoding | ~3s | 0 (local LaBSE) |
| RGCN Inference | <1s | 0 (local model) |
| **Total** | **~90s** | **~45** |

---

## 🔧 Key Technical Implementations

### 1. 🕷️ Scraper Upgrade — Zero-Cost Twitter Access

Bypasses the **$100/month Twitter API** using [Scweet](https://pypi.org/project/Scweet/) v5.3, a GraphQL-based scraper.

| Feature | Implementation |
|---|---|
| **Authentication** | Browser cookie (`TWITTER_AUTH_TOKEN`) — no API keys needed |
| **Proxy Support** | Optional `PROXY_URL` routing via `ScweetConfig(proxies={...})` |
| **Rate-Limit Resilience** | Catches `RateLimitError` / HTTP 429; falls back to profile-only features instead of crashing |
| **Delay Control** | Configurable `SCRAPE_DELAY_SECONDS` (default 3.0s) between API calls |

```python
# Rate-limit resilient tweet fetching
try:
    tweets = await client.aget_profile_tweets([username], limit=20)
except Exception as e:
    if _is_rate_limit_error(e):  # RateLimitError or HTTP 429
        logger.warning(f"⚠ Rate-limited on @{username}. Profile-only mode.")
        return []  # Node enters graph with zero tweet embedding
```

### 2. 📐 Feature Engineering — Tweet Embedding Strategy

**Key design:** The production pipeline uses a **norm-stabilized** LaBSE encoding to make predictions robust to the variable number of tweets Scweet fetches per account.

| Step | Operation | Purpose |
|---|---|---|
| 1 | `pooler_output` from LaBSE | Raw 768-D CLS-token embedding per tweet |
| 2 | `mean()` across all tweets | Tweet-count invariant (unlike sum) |
| 3 | L2-normalize to unit vector | Removes magnitude variation |
| 4 | Re-scale to target norm (~91.0) | Matches training-time magnitude (5 tweets × 18.2 norm) |

> **Why not raw sum?** The MGTAB training data used summed embeddings (~5 tweets/user), but in production, Scweet fetches 5–20 tweets depending on rate limits. Raw sum would produce vectors 4× larger for 20-tweet users. Mean-pool + fixed rescale ensures consistent magnitude regardless of tweet count.

### 3. 🕸️ Smart Graph Builder

PyTorch Geometric constructs a mini ego-graph with **7 relation types** following exact paper semantics:

| Relation | Type | Direction | Discovery Method |
|---|---|---|---|
| Follower | Explicit | neighbor → target | `aget_followers()` API |
| Friend | Explicit | target → neighbor | `aget_following()` API |
| Mention | Explicit | target → neighbor | Regex `@username` in tweets |
| Reply | Explicit | target → neighbor | GraphQL `in_reply_to_screen_name` |
| Quote | Explicit | target → neighbor | Tweet URL pattern matching |
| URL | Implicit | bidirectional ↔ | URL co-occurrence in tweets |
| Hashtag | Implicit | bidirectional ↔ | Hashtag co-occurrence in tweets |

> **Design Decision:** Neighbors without real profile or tweet data are **excluded** from the graph. Zero-vector placeholder nodes corrupt the RGCN's mean aggregation and produce unreliable predictions.

---

## 🗄️ Redis Caching Layer

Prediction results are cached via **Upstash Redis** (serverless HTTPS REST) to skip redundant ~90s scrape cycles.

| Feature | Implementation |
|---|---|
| **Provider** | [Upstash Redis](https://upstash.com) — serverless, HTTPS-based (no TCP sockets) |
| **Key Format** | `prediction:{lowercase_handle}` |
| **TTL** | `REDIS_CACHE_TTL` seconds (default 3600 = 1 hour) |
| **Cache Hit** | Returns instantly via `cache_hit` SSE event — entire scrape pipeline skipped |
| **Force Refresh** | `?refresh=true` query param clears cache and runs fresh analysis |
| **Graceful Degradation** | All cache ops silently return `None`/`False` on error — app works without Redis |

```env
# Upstash Redis credentials (optional)
UPSTASH_REDIS_REST_URL=https://your-instance.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_token_here
# REDIS_CACHE_TTL=3600
```

---

## 🎯 Confidence Calibration

Two post-processing layers prevent misleading confidence scores on out-of-distribution inputs:

### 1. Probability Clamping (Sparse Graph Protection)

Live ego-graphs (~20 nodes) are much smaller than the training graph (10,199 nodes). Probabilities are clamped to **[5%, 95%]** on degenerate graphs (<5 nodes or <5 edges), with a quality warning shown to the user.

### 2. High-Follower Calibration (Celebrity OOD Correction)

Accounts with >1M followers have extreme follower/friend ratios never seen in training. A **human-prior blend** scales with follower count:

| Followers | Blend Weight | Effect |
|---|---|---|
| <1M | 0% | No correction |
| 1M | 30% | Light human prior |
| 50M+ | 85% | Strong correction (0.90 human / 0.10 bot prior) |

---

## 🎨 Frontend Architecture

Multi-page SPA built with **React 19 + Vite 8 + React Router 7**, featuring a premium dark-mode cosmic design with canvas-based shooting star animations.

### Pages

| Route | Page | Description |
|---|---|---|
| `/` | **HomePage** | Hero section + "How It Works" pipeline + model comparison cards |
| `/detect` | **DetectorPage** | Dual-mode bot detection (One-Click SSE + Manual fallback) |
| `/analytics` | **AnalyticsPage** | Dataset overview, GNN model comparison table, feature importance |

### Dual-Mode Detection

| Mode | Description |
|---|---|
| **⚡ One-Click** | Enter `@username` → 5-step SSE progress stepper → result card. Handles `cache_hit` for instant results |
| **🔧 Manual** | Enter profile data, tweets, relations manually — fallback when scraping is unavailable |

### Key Components

| Component | Purpose |
|---|---|
| `ShootingStars` | Full-page `<canvas>` animation (60fps comets + twinkling stars) |
| `Hero` | Landing hero with gradient accents and live model stats |
| `ModelStats` | Pipeline visualization + GNN comparison cards |
| `ResultCard` | Prediction display with probability bars and quality warnings |
| `Navbar` / `Footer` | Navigation + tech stack badges + authors |

---

## 📊 Model Performance

Trained on the **MGTAB dataset** — 10,199 expert-annotated Twitter accounts with 7 relation types.

| Metric | RGCN (Ours) | GraphSAGE | GAT | GCN |
|---|---|---|---|---|
| **Test Accuracy** | **88.23%** | 87.16% | 81.67% | 79.21% |
| **Bot Recall** | **90.29%** | 88.85% | 84.53% | 68.70% |
| Training Epochs | 200 | 200 | 200 | 200 |
| Hidden Dim | 256 | 256 | 256 | 256 |

<details>
<summary><b>📈 5-Seed Cross-Validation Results (Mean ± Std)</b></summary>

| Model | Accuracy | F1-Score | Bot Recall | ROC-AUC |
|---|---|---|---|---|
| **RGCN** | **87.63% ± 1.2** | **0.7958 ± 0.01** | **90.45% ± 1.6** | **0.9551 ± 0.004** |
| GraphSAGE | 86.94% ± 1.0 | 0.7889 ± 0.01 | 91.77% ± 1.5 | 0.9479 ± 0.005 |
| GCN | 77.10% ± 1.7 | 0.5837 ± 0.05 | 61.70% ± 11.2 | 0.8323 ± 0.014 |
| GAT | 75.69% ± 3.0 | 0.5731 ± 0.12 | 65.50% ± 19.8 | 0.8148 ± 0.062 |

*Seeds: 42, 123, 456, 789, 1024 · All models: Adam lr=0.001, dropout=0.5, weighted CrossEntropyLoss*

</details>

### Top Features by Gradient Attribution

| Rank | Feature | Mean Gradient | Insight |
|---|---|---|---|
| 🥇 | `default_profile_image` | 0.1609 | Bots rarely upload custom avatars |
| 🥈 | `description_length` | 0.1433 | Profile completeness indicator |
| 🥉 | `geo_enabled` | 0.1347 | Bots rarely enable geolocation |
| 4 | `has_url` | 0.1316 | URL presence signals authenticity |
| 5 | `listed_count` | 0.1283 | Community recognition signal |

---

## 🔢 Feature Engineering

### 788-Dimensional Node Feature Vector

```
┌──────────────────────────────────────────────────────────────┐
│ Dim 0-19: Profile Features (20-D)                            │
│ ┌──────────┬──────────┬──────────┬──────────┬──────────┐    │
│ │ Boolean  │ Numeric  │ Derived  │ Legacy   │ Temporal │    │
│ │ verified │ followers│ ff_ratio │ geo_en.  │ created  │    │
│ │ def_prof │ friends  │ sn_len   │ bg_color │ _at      │    │
│ │ def_img  │ listed   │ name_len │ sb_fill  │          │    │
│ │ has_url  │ statuses │ desc_len │ sb_bord  │          │    │
│ │ bg_img   │ favs     │          │ bg_url   │          │    │
│ └──────────┴──────────┴──────────┴──────────┴──────────┘    │
├──────────────────────────────────────────────────────────────┤
│ Dim 20-787: Tweet Embeddings (768-D LaBSE)                   │
│ ┌────────────────────────────────────────────────────────┐   │
│ │ LaBSE pooler_output — mean-pooled + norm-stabilized    │   │
│ │ (Language-agnostic BERT Sentence Embedding)             │   │
│ │ Supports 109+ languages • Re-scaled to target norm     │   │
│ └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Normalization Pipeline

```
Numerical Counts → log(1 + x) → MinMax Scale [0, 1]
Booleans → Direct 0.0 / 1.0 encoding
created_at → log(timestamp_ns) → MinMax Scale
Tweet Texts → LaBSE pooler_output → Mean-pool → L2-norm → Re-scale (×91.0)
```

---

## 📁 Project Structure

```
MGTAB/
├── 📂 frontend/                        # React 19 + Vite 8 + React Router 7
│   ├── src/
│   │   ├── api/predict.js              # API client (SSE + REST + cache_hit)
│   │   ├── components/
│   │   │   ├── Navbar.jsx              # Navigation with active route
│   │   │   ├── Hero.jsx / Hero.css     # Landing hero section
│   │   │   ├── ModelStats.jsx/css      # Pipeline + model comparison
│   │   │   ├── ShootingStars.jsx       # Canvas shooting star animation
│   │   │   ├── ResultCard.jsx/css      # Prediction result display
│   │   │   ├── ProfileForm.jsx         # 20-field profile form
│   │   │   ├── TweetInput.jsx          # Dynamic tweet input
│   │   │   ├── RelationsEditor.jsx     # Edge/relation editor
│   │   │   └── Footer.jsx/css          # Footer with tech badges
│   │   ├── pages/
│   │   │   ├── HomePage.jsx            # Hero + ModelStats
│   │   │   ├── DetectorPage.jsx/css    # Dual-mode detection UI
│   │   │   └── AnalyticsPage.jsx/css   # Dataset & model analytics
│   │   ├── App.jsx                     # Router + layout
│   │   └── index.css                   # Global design system
│   ├── package.json
│   └── vite.config.js
│
├── 📂 backend/                         # FastAPI + PyTorch
│   ├── app/
│   │   ├── main.py                     # FastAPI endpoints + SSE + cache
│   │   ├── cache.py                    # Upstash Redis caching layer
│   │   ├── scraper.py                  # Scweet v5.3 Twitter scraper
│   │   ├── features.py                # LaBSE + profile feature engineering
│   │   ├── graph_builder.py           # PyTorch Geometric graph construction
│   │   ├── inference.py               # RGCN + confidence calibration
│   │   ├── rgcn_model.py              # RGCN architecture (RGCNConv)
│   │   ├── normalization.py           # MinMax bounds from MGTAB dataset
│   │   └── config.py                  # Environment & model constants
│   ├── best_rgcn.pt                   # Trained RGCN weights (6.5 MB)
│   ├── Dockerfile                     # HuggingFace Spaces deployment
│   ├── requirements.txt
│   └── .env.example
│
└── 📂 Datasets and precrosessing/     # Training pipeline & evaluation
    ├── Dataset/                       # Raw MGTAB tensors
    │   ├── edge_index.pt              # Edge connectivity (27 MB)
    │   ├── edge_type.pt               # 7 relation type labels (13 MB)
    │   ├── features.pt                # Node features (32 MB)
    │   └── labels_bot.pt              # Bot/human labels
    ├── 1–5. Step - */                  # Data preprocessing pipeline
    │   ├── shape.py                   # Tensor shape verification
    │   ├── check_features.py          # NaN/Inf/zero checks
    │   ├── check_labels_count.py      # Class distribution analysis
    │   ├── contruct_graph.py          # Builds PyG Data → graph_data.pt
    │   └── split_data.py              # 80/10/10 train/val/test split
    ├── 6. Step - Models/              # GNN model training scripts
    │   ├── rgcn_model.py              # RGCN training (200 epochs)
    │   ├── gcn_model.py               # GCN baseline
    │   ├── gat_model.py               # GAT baseline (2-head)
    │   └── graphsage_model.py         # GraphSAGE baseline
    ├── 7. Step - models_saved/        # Trained model weights
    │   ├── best_rgcn.pt               # RGCN checkpoint (6.5 MB)
    │   ├── best_gcn.pt                # GCN checkpoint
    │   ├── best_gat.pt                # GAT checkpoint
    │   └── best_graphsage.pt          # GraphSAGE checkpoint
    ├── run_full_evaluation.py         # 5-seed × 4-model evaluation
    ├── run_ablation.py                # RGCN ablation study
    ├── run_explainability.py          # Gradient attribution analysis
    ├── graph_data.pt                  # Full MGTAB graph (10,199 nodes)
    └── *.csv                          # Evaluation result logs
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/predict/username/{handle}` | One-click SSE bot detection (with Redis cache) |
| `GET` | `/predict/username/{handle}?refresh=true` | Force fresh analysis (clears cache) |
| `POST` | `/predict/user` | Manual mode — send raw profile/tweet/relation data |
| `GET` | `/model/info` | Model metadata (architecture, accuracy, dataset) |
| `GET` | `/health` | Health check (`model_loaded` status) |
| `GET` | `/features/schema` | Feature definitions for frontend form |

### SSE Event Types

| Event | Payload | When |
|---|---|---|
| `progress` | `{step, status, message}` | Each pipeline stage starts |
| `cache_hit` | `{label_pred, prob_human, prob_bot, ...}` | Redis cache hit — instant result |
| `scrape_complete` | `{username, neighbors_found, ...}` | Scraping finished |
| `result` | `{label_pred, prob_human, prob_bot, confidence, graph_info}` | RGCN prediction complete |
| `error` | `{message, status_code}` | Any error (404, 429, 401, etc.) |
| `done` | `{status: "complete"}` | Stream finished |

---

## 🚀 Setup & Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| pip | Latest |

### 1. Clone the Repository

```bash
git clone https://github.com/Arihant0008/MGTAB.git
cd MGTAB
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install "numpy<2.0.0"
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric>=2.4.0
pip install transformers==4.44.0
pip install sentence-transformers==3.0.0
```

### 3. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required: Get from x.com → F12 → Application → Cookies → auth_token
TWITTER_AUTH_TOKEN=your_auth_token_here

# Optional: Proxy for rate-limit avoidance
# PROXY_URL=http://user:pass@host:port

# Optional: Delay between API calls (default: 3.0 seconds)
# SCRAPE_DELAY_SECONDS=3.0

# Optional: Upstash Redis for prediction caching (instant repeat lookups)
# UPSTASH_REDIS_REST_URL=https://your-instance.upstash.io
# UPSTASH_REDIS_REST_TOKEN=your_token_here
# REDIS_CACHE_TTL=3600
```

### 4. Start the Backend

```bash
uvicorn app.main:app --reload
```

> ⚠️ **First Run:** The backend will download the **LaBSE model (~1.8 GB)** on the first prediction request. Subsequent runs use the cached model.

### 5. Frontend Setup

```bash
cd ../frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) and enter a Twitter handle to analyze.

---

## 🐳 Deployment

### Backend → Hugging Face Spaces (Docker)

The Dockerfile uses **CPU-only PyTorch** to reduce the image size from ~3 GB to ~1.5 GB.

```bash
# Push to your HF Space
cd backend
git clone https://huggingface.co/spaces/YOUR_USERNAME/mgtab-bot-detector hf_deploy
cp -r app Dockerfile requirements.txt best_rgcn.pt .env.example hf_deploy/
cd hf_deploy
git add . && git commit -m "Deploy" && git push
```

Then add these **Repository Secrets** in your Space settings:
- `TWITTER_AUTH_TOKEN` (required)
- `UPSTASH_REDIS_REST_URL` (optional — enables caching)
- `UPSTASH_REDIS_REST_TOKEN` (optional — enables caching)

### Frontend → Vercel

```bash
cd frontend

# Update API URL in src/api/predict.js:
# const API_BASE = 'https://YOUR_USERNAME-mgtab-bot-detector.hf.space';

npm run build   # Verify build
# Push to GitHub → Import in Vercel → Auto-deploy
```

### Live Deployment

| Service | Platform | URL |
|---|---|---|
| Frontend UI | ▲ Vercel | [`mgtab.me`](https://www.mgtab.me/) |
| Backend API | 🤗 Hugging Face Spaces | [`arihant0008-mgtab-bot-detector-main.hf.space`](https://arihant0008-mgtab-bot-detector-main.hf.space) |
| Cache Layer | Upstash Redis | Serverless REST (auto-provisioned) |

---

## ⚠️ Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Twitter rate limits (HTTP 429) | Can blind the 768-D text features for some neighbors | Graceful fallback to profile-only features + retry with 5s back-off |
| Live ego-graphs are small (~20 nodes) | Training graph has 10,199 nodes; inference graph is much smaller | Self-loop fallback + probability clamping [5%, 95%] |
| Cookie-based auth expires | `auth_token` lasts 1–2 years but may need refresh | Set as HF Secret; monitor `/health` endpoint |
| Class imbalance (2.3:1 human:bot) | Slight bias toward human classification | Corrected by proper embedding normalization |
| Celebrity accounts (>1M followers) | Extreme follower ratios are out-of-distribution | High-follower calibration with human-prior blend |

### Future Work

- 🔄 **Managed Scraper Migration** — Replace cookie-based Scweet with a fully managed [Apify Actor](https://apify.com/) for zero-maintenance scraping
- 📦 **Batch Inference** — Support CSV upload of multiple usernames for bulk analysis
- 🌐 **Real-Time Graph Expansion** — Dynamic ego-graph growth with streaming neighbor discovery
- 🧪 **Adversarial Robustness** — Test against LLM-powered bot accounts and adversarial profile manipulation
- 📊 **Explainability Dashboard** — Visualize which features and graph structures drive the prediction
- 🔔 **Webhook Notifications** — Push alerts when a cached prediction expires or a watched account changes status

### ✅ Recently Completed

- ✅ **Redis Caching** — Upstash REST-based prediction caching for instant repeat lookups
- ✅ **Confidence Calibration** — Probability clamping + high-follower OOD correction
- ✅ **Multi-Page Frontend** — React Router SPA with Home, Detector, and Analytics pages
- ✅ **Dual-Mode Detection** — One-Click SSE + Manual Mode fallback
- ✅ **Analytics Dashboard** — Dataset overview, model comparison, feature importance
- ✅ **Premium UI** — Shooting stars animation, glassmorphism cards, dark cosmic theme
- ✅ **Custom Domain** — Live at [mgtab.me](https://www.mgtab.me/)

---

## 📚 References

1. **Shi, S., Qiao, K., Chen, J., et al.** (2023). *MGTAB: A Multi-Relational Graph-Based Twitter Account Detection Benchmark.* arXiv:2301.12174. [[Paper]](https://arxiv.org/abs/2301.12174) [[Dataset]](https://github.com/GraphDetec/MGTAB)

2. **Schlichtkrull, M., et al.** (2018). *Modeling Relational Data with Graph Convolutional Networks.* ESWC 2018. [[Paper]](https://arxiv.org/abs/1703.06103)

3. **Feng, F., Yang, Y., et al.** (2022). *Language-agnostic BERT Sentence Embedding (LaBSE).* [[Model]](https://huggingface.co/sentence-transformers/LaBSE)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by Arihant, Aayush & Pratham — B.E. Final Year Project 2025–26**

*If this project helped you, consider giving it a ⭐*

[![GitHub Stars](https://img.shields.io/github/stars/Arihant0008/MGTAB?style=social)](https://github.com/Arihant0008/MGTAB)

</div>


<div align="center">

# 🤖 MGTAB Bot Detector

### Multi-Relational Graph-Based Twitter/X Account Bot Detection

[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
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
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Project Structure](#-project-structure)
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
    subgraph Frontend ["🖥️ Frontend — React 19 + Vite"]
        UI["Search Bar<br/>@username Input"]
        Stepper["5-Step SSE<br/>Progress Stepper"]
        Result["Prediction Card<br/>Human / Bot"]
    end

    subgraph Backend ["⚙️ Backend — FastAPI + Python 3.11"]
        API["FastAPI Router<br/>SSE Endpoint"]
        
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
        end
    end

    subgraph External ["🌐 External Services"]
        Twitter["Twitter/X<br/>GraphQL API"]
        HF["🤗 Hugging Face<br/>Spaces (Docker)"]
        Vercel["▲ Vercel<br/>CDN"]
    end

    UI -->|"GET /predict/username/@handle"| API
    API --> Auth
    Auth -->|"Cookie + Proxy"| Twitter
    Auth --> Profile
    Profile --> Ego
    Ego --> Enrich
    Enrich --> Adapter
    Adapter --> LaBSE
    LaBSE --> GraphB
    GraphB --> RGCN
    RGCN -->|"SSE Stream"| Stepper
    Stepper --> Result
    
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

### 2. 📐 Feature Engineering Math Fix

**The breakthrough:** Discovered that the original MGTAB training data used **raw summed** LaBSE `pooler_output` — not L2-normalized, not averaged.

| Metric | Before Fix (Broken) | After Fix (Correct) |
|---|---|---|
| LaBSE Normalization | L2-normalized (unit vectors) | Raw pooler_output |
| Aggregation | `mean()` across tweets | `sum()` across tweets |
| Embedding Norm | ~0.50 | ~112–135 |
| Training Data Norm | — | ~18.9 (matched ✅) |
| @shipinbot Detection | ❌ Human 58.8% | ✅ **Bot 100%** |
| @NaiTUbot Detection | ❌ Human 66.4% | ✅ **Bot 100%** |

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

## 📊 Model Performance

Trained on the **MGTAB dataset** — 10,199 expert-annotated Twitter accounts with 7 relation types.

| Metric | RGCN (Ours) | GCN | GAT | GraphSAGE |
|---|---|---|---|---|
| **Test Accuracy** | **88.23%** | 84.51% | 85.12% | 83.97% |
| **Bot Recall** | **90.29%** | 86.44% | 87.01% | 85.22% |
| **F1-Score** | **0.8791** | 0.8389 | 0.8456 | 0.8341 |
| Training Epochs | 200 | 200 | 200 | 200 |
| Hidden Dim | 256 | 256 | 256 | 256 |

### Top Features by Information Gain

| Rank | Feature | Importance |
|---|---|---|
| 🥇 | `followers_friends_ratio` | Highest discriminative power |
| 🥈 | `statuses_count` | Tweet volume anomaly detection |
| 🥉 | `listed_count` | Community recognition signal |
| 4 | `description_length` | Profile completeness indicator |
| 5 | `favourites_count` | Engagement authenticity metric |

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
│ │ Summed pooler_output from LaBSE                        │   │
│ │ (Language-agnostic BERT Sentence Embedding)             │   │
│ │ Supports 109+ languages • Raw output, no L2 norm       │   │
│ └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Normalization Pipeline

```
Numerical Counts → log(1 + x) → MinMax Scale [0, 1]
Booleans → Direct 0.0 / 1.0 encoding
created_at → log(timestamp_ns) → MinMax Scale
Tweet Texts → LaBSE pooler_output → Sum aggregation
```

---

## 📁 Project Structure

```
MGTAB/
├── 📂 frontend/                    # React 19 + Vite
│   ├── src/
│   │   ├── api/predict.js          # API client (SSE + REST)
│   │   ├── pages/DetectorPage.jsx  # Main detection UI
│   │   └── pages/DetectorPage.css  # Styling
│   ├── package.json
│   └── vite.config.js
│
├── 📂 backend/                     # FastAPI + PyTorch
│   ├── app/
│   │   ├── main.py                 # FastAPI endpoints + SSE
│   │   ├── scraper.py              # Scweet v5.3 Twitter scraper
│   │   ├── features.py             # LaBSE + profile feature engineering
│   │   ├── graph_builder.py        # PyTorch Geometric graph construction
│   │   ├── inference.py            # RGCN model loading + prediction
│   │   ├── rgcn_model.py           # RGCN architecture (RGCNConv)
│   │   ├── normalization.py        # MinMax bounds from MGTAB dataset
│   │   └── config.py               # Environment & model constants
│   ├── best_rgcn.pt                # Trained RGCN weights (6.5 MB)
│   ├── Dockerfile                  # HuggingFace Spaces deployment
│   ├── requirements.txt
│   └── .env.example
│
└── 📂 Datasets and precrosessing/  # Training pipeline
    ├── graph_data.pt               # Full MGTAB graph (10,199 nodes)
    └── best_rgcn.pt                # Model checkpoint
```

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
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric>=2.4.0
pip install transformers==4.44.0
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

Then add `TWITTER_AUTH_TOKEN` as a **Repository Secret** in your Space settings.

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
| Backend API | 🤗 Hugging Face Spaces | [`arihant0008-mgtab-bot-detector-main.hf.space`](https://arihant0008-mgtab-bot-detector-main.hf.space) |
| Frontend UI | ▲ Vercel | *Deploy from GitHub* |

---

## ⚠️ Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Twitter rate limits (HTTP 429) | Can blind the 768-D text features for some neighbors | Graceful fallback to profile-only features |
| Live ego-graphs are small (~20 nodes) | Training graph has 10,199 nodes; inference graph is much smaller | Self-loop fallback for feature-only mode |
| Cookie-based auth expires | `auth_token` lasts 1–2 years but may need refresh | Set as HF Secret; monitor `/health` endpoint |
| Class imbalance (2.3:1 human:bot) | Slight bias toward human classification | Corrected by proper embedding normalization |

### Future Work

- 🔄 **Managed Scraper Migration** — Replace cookie-based Scweet with a fully managed [Apify Actor](https://apify.com/) for zero-maintenance scraping
- 📦 **Batch Inference** — Support CSV upload of multiple usernames for bulk analysis
- 🌐 **Real-Time Graph Expansion** — Dynamic ego-graph growth with streaming neighbor discovery
- 🧪 **Adversarial Robustness** — Test against LLM-powered bot accounts and adversarial profile manipulation
- 📊 **Explainability Dashboard** — Visualize which features and graph structures drive the prediction

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

**Built with ❤️ as a Final Year Engineering Project**

*If this project helped you, consider giving it a ⭐*

[![GitHub Stars](https://img.shields.io/github/stars/Arihant0008/MGTAB?style=social)](https://github.com/Arihant0008/MGTAB)

</div>

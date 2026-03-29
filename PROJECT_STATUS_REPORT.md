# MGTAB Project — Senior Project Manager Status Report

**Date:** 29 March 2026  
**Project:** MGTAB — Multi-relational Graph-based Twitter Account Bot Detection  
**Prepared by:** Senior Project Manager  

---

## 1. Executive Summary

The MGTAB project aims to detect bot accounts on Twitter/X using Graph Neural Networks (GNNs) on the MGTAB benchmark dataset. As of today, the **entire ML/Data-Science pipeline is complete** — from raw data validation through model training to final evaluation. We have trained and benchmarked **four GNN architectures** and identified **RGCN** as the best-performing model. 

**However**, the project currently exists as a collection of standalone Python scripts. There is **no user interface, no backend API, and no deployment infrastructure**. The next phase of the project must focus on productionizing the trained model into a usable application.

---

## 2. What Has Been Completed ✅

### Phase 1: Data Validation & Preprocessing

| Step | Script | Purpose | Status |
|------|--------|---------|--------|
| 1 | `shape.py` | Verified tensor shapes of all 6 dataset files (`edge_index`, `edge_type`, `edge_weight`, `features`, `labels_bot`, `labels_stance`) | ✅ Done |
| 2 | `check_features.py` | Checked for NaN, Inf, zero-rows in features; validated edge index bounds; verified edge type ranges and edge weight integrity | ✅ Done |
| 3 | `check_labels_count.py` | Analysed class distribution — counted bot vs. human labels and computed class imbalance ratios (~2.7:1 imbalance) | ✅ Done |

### Phase 2: Graph Construction

| Step | Script | Purpose | Status |
|------|--------|---------|--------|
| 4a | `contruct_graph.py` | Built a PyTorch Geometric `Data` object combining features, edge_index, edge_attr (type + weight), and labels → saved as `graph_data.pt` (~73 MB) | ✅ Done |
| 4b | `verify_graph.py` | Verified the constructed graph's node features, edge shapes, edge attributes, and label shapes | ✅ Done |

### Phase 3: Train / Validation / Test Split

| Step | Script | Purpose | Status |
|------|--------|---------|--------|
| 5a | `split_data.py` | Created random permutation-based masks → 80% train / 10% val / 10% test; saved masks back into `graph_data.pt` | ✅ Done |
| 5b | `check.py` | Verified node counts per split (printed train/val/test counts) | ✅ Done |

### Phase 4: Model Training & Evaluation

Four GNN architectures trained for **200 epochs** each with:
- **Optimizer:** Adam (lr = 0.001)
- **Class imbalance handling:** Weighted CrossEntropyLoss (bot weight ≈ 2.7×)
- **Dropout:** 0.5
- **Best model checkpoint:** Saved based on validation accuracy
- **Logging:** Per-epoch CSV logs (loss, accuracy, recall, time) in `logs/`

| Model | Architecture | Hidden Dim | Heads | Test Accuracy | Bot Recall | Checkpoint |
|-------|-------------|-----------|-------|---------------|------------|------------|
| GCN | `GCNConv` → 256 → classes | 256 | — | 79.21% | 68.70% | `best_gcn.pt` |
| GAT | `GATConv` → 128×2 heads → classes | 256 (128×2) | 2 | 81.67% | 84.53% | `best_gat.pt` |
| GraphSAGE | `SAGEConv` → 256 → classes | 256 | — | 87.16% | 88.85% | `best_graphsage.pt` |
| **RGCN** ⭐ | `RGCNConv` → 256 → classes | 256 | — | **88.23%** | **90.29%** | `best_rgcn.pt` |

### Phase 5: Post-Training Analytics

| Script | Purpose | Status |
|--------|---------|--------|
| `confusion_matrix.py` | Generates confusion matrix & sklearn classification report for the best RGCN model on test set | ✅ Done |
| `model_comparision.py` | Compares all 4 models side-by-side; exports `final_model_results.csv` | ✅ Done |

### Key Artifacts Produced

```
MGTAB/
├── Dataset/                    # Raw tensors (features, edges, labels)
├── graph_data.pt               # Constructed PyG graph with train/val/test masks (73 MB)
├── best_rgcn.pt                # Best model checkpoint (~6.5 MB)
├── 7. Step - models_saved/     # All model checkpoints (GCN, GAT, GraphSAGE)
├── logs/                       # 4 training log CSVs
├── final_model_results.csv     # Model comparison table
└── confusion_matrix.py         # Evaluation script
```

---

## 3. What Has NOT Been Built Yet ❌

There is **zero** user-facing or backend infrastructure. The entire project is a set of offline training scripts. Below is the comprehensive breakdown of what must be built.

---

## 4. What Needs to Be Done — Backend

### 4.1 Inference API (Flask / FastAPI)

Build a REST API that:
- Loads the best trained model (`best_rgcn.pt`) at startup
- Accepts user profile features as input (JSON payload)
- Runs inference on the graph and returns bot/human prediction with confidence score
- Endpoints needed:
  - `POST /api/predict` — Single account prediction
  - `POST /api/predict/batch` — Batch prediction for multiple accounts
  - `GET /api/model/info` — Returns model metadata (architecture, accuracy, version)
  - `GET /api/health` — Health check

### 4.2 Data Pipeline / Feature Extraction Module

Build a module that:
- Accepts a raw Twitter/X username or profile data
- Extracts the same features used during training (the 768-dim or similar feature vector)
- Converts raw profile/tweet data into the tensor format the model expects
- Handles missing fields gracefully

### 4.3 Graph Integration Layer

The current model operates on a static graph. For production:
- Decide strategy: **Static graph lookup** (check if user exists in the prebuilt graph) or **Dynamic graph augmentation** (add new user nodes to the graph at inference time)
- If dynamic: implement neighbor sampling and edge construction for new nodes
- If static: implement a node-ID lookup service from username → graph index

### 4.4 Database Layer

- **User Results DB** (PostgreSQL / SQLite): Store past predictions, user queries, timestamps
- **Model Registry**: Track model versions, training metrics, and deployment history
- **Session / Auth Store**: If user accounts are needed

### 4.5 Authentication & Rate Limiting

- API key-based or JWT-based auth for the prediction endpoints
- Rate limiting to prevent abuse
- CORS configuration for frontend access

### 4.6 Logging & Monitoring

- Structured logging for all API requests and predictions
- Error tracking (e.g., Sentry)
- Model performance monitoring (drift detection over time)

---

## 5. What Needs to Be Done — Frontend / User Interface

### 5.1 Landing Page / Dashboard

A modern, responsive web UI that includes:
- **Hero section**: Project name, brief description, CTA to check an account
- **How it works**: Visual explanation of the GNN-based detection pipeline
- **Model performance cards**: Display accuracy, recall, F1 from all 4 models
- **Live demo section**: Input field to check a Twitter/X account

### 5.2 Account Checker Page

The core interactive feature:
- **Input form**: Enter a Twitter/X username or paste profile URL
- **Result card**: Shows bot/human prediction, confidence score (%), risk level indicator (Low / Medium / High)
- **Explanation panel**: Why the model flagged this account (feature importance, neighbor analysis)
- **History sidebar**: Past queries and results (if logged in)

### 5.3 Analytics Dashboard

For power users / researchers:
- **Model comparison chart**: Bar/radar chart comparing GCN, GAT, GraphSAGE, RGCN
- **Training curves**: Interactive line charts from the training CSVs (loss, accuracy, recall over epochs)
- **Confusion matrix heatmap**: Visual rendering of the RGCN confusion matrix
- **Dataset statistics**: Node count, edge count, class distribution pie chart

### 5.4 Batch Analysis Page

- Upload a CSV of usernames
- Get bulk predictions
- Download results as CSV/PDF report

### 5.5 About / Documentation Page

- Project methodology, dataset reference (MGTAB paper), model architecture diagrams
- API documentation (Swagger-style or embedded docs)
- Team/credits section

### 5.6 Technology Recommendations for Frontend

| Concern | Recommendation |
|---------|---------------|
| Framework | **Next.js** or **Vite + React** |
| Styling | **Tailwind CSS** or **Vanilla CSS** with dark mode |
| Charts | **Recharts** or **Chart.js** for training curves & comparisons |
| State | **React Context** or **Zustand** (lightweight) |
| HTTP Client | **Axios** or **fetch** |
| Animations | **Framer Motion** for micro-interactions |

---

## 6. Recommended Next Steps (Priority Order)

| # | Task | Priority | Estimated Effort |
|---|------|----------|-----------------|
| 1 | Build FastAPI inference API with `/predict` endpoint | 🔴 Critical | 2–3 days |
| 2 | Build feature extraction module for raw profile data | 🔴 Critical | 2–3 days |
| 3 | Design & implement the frontend landing + checker page | 🟡 High | 3–4 days |
| 4 | Integrate frontend ↔ backend API calls | 🟡 High | 1–2 days |
| 5 | Build analytics dashboard with charts | 🟢 Medium | 2–3 days |
| 6 | Add database layer for storing predictions | 🟢 Medium | 1–2 days |
| 7 | Add batch analysis page | 🔵 Low | 1–2 days |
| 8 | Add auth, rate limiting, and deployment (Docker) | 🔵 Low | 2–3 days |
| 9 | Documentation & API docs page | 🔵 Low | 1 day |

**Total estimated effort: ~15–23 working days (3–5 weeks)**

---

## 7. Architecture Diagram (Proposed)

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React/Next.js)                │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Landing  │  │ Account      │  │ Analytics Dashboard   │  │
│  │ Page     │  │ Checker      │  │ (Charts, Confusion    │  │
│  │          │  │ (Input/Result│  │  Matrix, Comparisons) │  │
│  └──────────┘  └───────┬──────┘  └───────────────────────┘  │
│                        │                                     │
└────────────────────────┼─────────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  BACKEND (FastAPI / Flask)                   │
│  ┌────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│  │ /predict   │  │ /predict/batch│  │ /model/info        │  │
│  │ endpoint   │  │ endpoint      │  │ endpoint           │  │
│  └─────┬──────┘  └───────┬───────┘  └────────────────────┘  │
│        │                 │                                   │
│  ┌─────▼─────────────────▼──────┐                           │
│  │  Feature Extraction Module   │                           │
│  │  (Raw data → tensor)        │                           │
│  └─────────────┬────────────────┘                           │
│                │                                             │
│  ┌─────────────▼────────────────┐                           │
│  │  RGCN Inference Engine       │                           │
│  │  (best_rgcn.pt + graph_data) │                           │
│  └──────────────────────────────┘                           │
│                                                              │
│  ┌──────────────────────────────┐                           │
│  │  Database (PostgreSQL/SQLite)│                           │
│  │  - Prediction history        │                           │
│  │  - Model registry            │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Risks & Considerations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Static graph — new users not in graph | Cannot predict for unknown users | Implement dynamic node insertion or fallback to feature-only classifier |
| Model overfitting (train acc significantly > test acc on some models) | Inaccurate production predictions | Use RGCN (smallest gap), add regularization |
| No Twitter/X API access | Cannot fetch live profile data | Allow manual feature input or use cached dataset |
| Class imbalance (~2.7:1) | Biased predictions | Already addressed with weighted loss; monitor in production |
| Large graph file (73 MB) | Slow cold starts | Pre-load graph at API startup; use model caching |

---

*This report was generated based on a complete code review of the MGTAB project repository as of 29 March 2026.*

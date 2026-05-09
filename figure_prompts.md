# Figure Generation Prompts — MGTAB Thesis

> Use these prompts with any AI image generator (DALL-E, Midjourney, etc.) or as specs for draw.io/matplotlib.
> For charts (6.1–6.6), use matplotlib/Excel with the actual data provided.

---

## Chapter 1

**Figure 1.1 — Evolution of Twitter Bot Sophistication**

```
A clean academic timeline diagram on white background showing the evolution of Twitter bots across three eras. Left section labeled "2010-2015: First-Gen Bots" with icons of egg avatar, repetitive tweets, 0 followers. Middle section "2016-2020: Social Spambots" with realistic profile photo, varied posting schedule, normal follower counts. Right section "2021-Present: LLM-Powered Bots" with AI-generated avatar, GPT-written content, organic engagement patterns. Arrow along bottom showing increasing sophistication. Professional technical diagram style, no decorations, grayscale with blue accent color. Suitable for an engineering thesis.
```

**Figure 1.2 — Isolated vs. Graph-Based Account Analysis**

```
A side-by-side comparison diagram for an academic thesis. LEFT side labeled "Traditional: Isolated Analysis" shows a single Twitter profile card with metadata fields (followers: 500, tweets: 3000, age: 2 years) and a question mark verdict. RIGHT side labeled "Graph-Based Analysis" shows the same profile card at center connected by colored lines to 8 surrounding smaller profile nodes, with lines labeled "follows", "mentions", "replies", forming a small network graph with a green checkmark verdict. White background, clean flat design, blue and gray color scheme, professional engineering diagram style.
```

---

## Chapter 3

**Figure 3.1 — High-Level System Architecture**

```
A professional three-tier system architecture diagram for a thesis. Top tier labeled "Frontend — React 19 + Vite" contains three boxes: HomePage, DetectorPage, AnalyticsPage, connected to an "API Client (predict.js)" box. Middle tier labeled "Backend — FastAPI + Python 3.11" contains boxes: FastAPI Router, Scweet Scraper, Feature Engineering, Graph Builder, RGCN Inference, connected by arrows showing data flow. Bottom tier labeled "External Services" shows Twitter/X GraphQL API, Hugging Face Spaces, and Vercel CDN icons. Arrows show SSE stream from backend to frontend, HTTP requests from scraper to Twitter. White background, rectangular boxes with rounded corners, blue/teal color palette, clean engineering diagram.
```

**Figure 3.2 — React Component Hierarchy Diagram**

```
A tree-structured component hierarchy diagram on white background. Root node "App.jsx (BrowserRouter)" branches into "Navbar.jsx" and three route branches. Route "/" leads to "HomePage.jsx" which contains "Hero.jsx" and "ModelStats.jsx". Route "/detect" leads to "DetectorPage.jsx" which contains four children: "ProfileForm.jsx", "TweetInput.jsx", "RelationsEditor.jsx", "ResultCard.jsx". Route "/analytics" leads to "AnalyticsPage.jsx". Each node is a rounded rectangle with filename. Lines connect parent to children. Flat design, monospace font for filenames, blue color scheme, academic diagram style.
```

**Figure 3.3 — SSE Data Flow Sequence Diagram**

```
A UML sequence diagram on white background with 4 vertical lifelines labeled: "User/Browser", "React Frontend", "FastAPI Backend", "Twitter API". Arrows show: User types username and clicks Analyze, Frontend sends GET /predict/username/handle to Backend, Backend sends multiple requests to Twitter API (scrape profile, fetch tweets, fetch followers), Backend sends SSE events back to Frontend (progress step 1, progress step 2, progress step 3, progress step 4, scrape_complete, progress step 5, result), Frontend displays result to User. Dashed arrows for responses, solid for requests. Professional black and white with blue accent lines, engineering thesis style.
```

**Figure 3.4 — FastAPI Route Structure**

```
A table-style diagram showing 5 API endpoints on white background. Each row shows: HTTP method badge (GET/POST in colored pill), route path, and description. Routes are: POST /predict/user (Manual JSON prediction), GET /predict/username/{handle} (SSE one-click analysis), GET /model/info (Model metadata), GET /health (Health check), GET /features/schema (Feature definitions). Clean tabular layout with alternating light gray rows, monospace font for routes, blue POST badge, green GET badges. Academic technical diagram style.
```

**Figure 3.5 — Deployment Architecture (Vercel + Hugging Face Spaces)**

```
A deployment infrastructure diagram on white background. Left side shows "User Browser" connecting via HTTPS to "Vercel CDN" cloud shape containing "React Static Build (index.html, JS, CSS)". Right side shows "Hugging Face Spaces" cloud shape containing a Docker container icon with "FastAPI + PyTorch + LaBSE" inside and "best_rgcn.pt (6.5MB)" file icon. An arrow labeled "API Calls (HTTPS)" connects Vercel to HF Spaces. Below HF Spaces, an arrow labeled "GraphQL Requests" connects to "Twitter/X API" icon. Domain "mgtab.me" labeled above Vercel. Clean cloud architecture diagram, blue and orange color scheme.
```

---

## Chapter 4

**Figure 4.1 — Ego-Graph Scraping Pipeline Flowchart**

```
A vertical flowchart on white background showing 4 sequential steps with boxes and arrows. Step 1 "Scrape Target": boxes for "Fetch Profile (aget_user_info)" → "Check if Private?" diamond (Yes→Error, No→continue) → "Fetch 20 Tweets" → "Extract URLs, Hashtags". Step 2 "Discover Neighbors": parallel boxes for "Fetch 10 Followers", "Fetch 10 Following", "Extract Mentions", "Extract Replies", "Extract Quotes" all flowing into "Deduplicate (~50 unique)". Step 3 "Enrich Neighbors": loop box "For each neighbor: Fetch Profile + 5 Tweets, Track URLs & Hashtags". Step 4 "Build Relations": boxes for "Create directed edges" → "Add URL co-occurrence edges" → "Add Hashtag co-occurrence edges" → "Return request_data". Blue header boxes for each step, white process boxes, yellow diamond for decision. Clean engineering flowchart style.
```

**Figure 4.2 — Rate-Limit Fallback Decision Flow**

```
A simple decision flowchart on white background. Start box "Fetch tweets for neighbor @user" → diamond "Rate Limited? (HTTP 429)" → Yes branch: "Log warning" → "Return empty tweet list" → "Node enters graph with profile features only (tweet embedding = zero vector)" in orange box. No branch: "Return tweet list" → "Node enters graph with full 788-dim features" in green box. Both paths merge at "Continue to next neighbor". Clean flowchart style, green for success path, orange for fallback path, white background, suitable for thesis.
```

**Figure 4.3 — 788-Dimensional Feature Vector Layout**

```
A horizontal stacked bar diagram on white background showing the 788-dimensional feature vector. Left segment (narrow, ~2.5% width) colored blue labeled "Profile Features (20-D)" with sub-labels: "10 Boolean, 5 Numerical, 5 Derived". Right segment (wide, ~97.5% width) colored teal labeled "LaBSE Tweet Embedding (768-D)" with sub-label "Summed pooler_output, raw (no L2 norm)". Dimension indices marked: [0-19] under blue segment, [20-787] under teal segment. Total labeled "788 dimensions" above. Clean flat design, white background, engineering diagram style.
```

**Figure 4.4 — Log-MinMax Normalization Pipeline**

```
A horizontal pipeline diagram on white background showing normalization steps. Input box "Raw Value (e.g., followers_count = 1500)" → arrow → box "Step 1: Log Transform: log(1 + 1500) = 7.31" → arrow → box "Step 2: MinMax Scale: (7.31 - 0.0) / (25.57 - 0.0) = 0.286" → arrow → output box "Normalized Value: 0.286 ∈ [0, 1]". Below, a note: "Boolean features skip this pipeline: direct 0.0/1.0 encoding". Clean left-to-right flow, light blue boxes, white background, academic style.
```

**Figure 4.5 — LaBSE Encoding and Summation Pipeline**

```
A vertical pipeline diagram on white background. Input: stack of 3 text boxes showing tweets "Great weather today!", "Just read a book", "Love coffee". Arrow down to "LaBSE Tokenizer (max 128 tokens, padding, truncation)". Arrow down to "LaBSE Model (pre-trained, 109 languages)". Arrow down to 3 horizontal vectors labeled "Tweet 1: [0.12, -0.3, 0.8, ...] (768-D)", "Tweet 2: [0.05, 0.2, -0.1, ...]", "Tweet 3: [0.3, -0.5, 0.4, ...]". Arrow down with "SUM (not average)" label in bold red to single vector "Summed embedding: [0.47, -0.6, 1.1, ...] (768-D)". Note: "Norm ≈ 18-20 (matching training data)". Clean academic diagram.
```

**Figure 4.6 — Seven Relation Types in the MGTAB Graph**

```
A diagram on white background showing a central node labeled "Target User" connected to surrounding nodes by 7 different colored/styled lines. Each line is labeled: (0) Follower — dashed blue arrow pointing TO target, (1) Friend — solid blue arrow pointing FROM target, (2) Mention — green arrow FROM target, (3) Reply — orange arrow FROM target, (4) Quoted — purple arrow FROM target, (5) URL — red bidirectional line, (6) Hashtag — yellow bidirectional line. Legend on the right listing all 7 types with their colors. 5 labeled "Directed" and 2 labeled "Undirected". Clean network diagram, white background, academic style.
```

**Figure 4.7 — Example Mini Ego-Graph for Inference**

```
A small network graph diagram on white background. Center node (large, blue) labeled "Target @elonmusk (Node 0, 788-D features)". Four surrounding nodes (smaller, gray) labeled "@user1 (Node 1)", "@user2 (Node 2)", "@user3 (Node 3)", "@user4 (Node 4)". Edges: @user1→Target labeled "R0: follower", Target→@user2 labeled "R1: friend", Target→@user3 labeled "R2: mention", Target↔@user2 labeled "R6: hashtag" (bidirectional). Each node shows a small feature vector icon "[788-D]". Clean graph visualization, blue center node, gray peripheral nodes, labeled colored edges, thesis style.
```

---

## Chapter 5

**Figure 5.1 — Standard GCN Message-Passing (Single Relation)**

```
A diagram on white background illustrating GCN message passing. Center node colored red labeled "Node i" surrounded by 4 neighbor nodes in gray labeled "Node j1, j2, j3, j4". Arrows from each neighbor point to center node, each labeled "same W". Below, the equation: h_i^(l+1) = σ(W · mean(h_j for j in neighbors) + W_0 · h_i). Caption: "All edges use the SAME weight matrix W — no relation-type distinction". Clean mathematical diagram style, white background, suitable for academic thesis.
```

**Figure 5.2 — RGCN Message-Passing with Relation-Specific Weights**

```
A diagram on white background illustrating RGCN message passing. Center node colored red labeled "Node i (target)". Surrounding neighbors grouped by relation type: 2 blue nodes labeled "Followers" with arrows labeled "W_follower", 2 green nodes labeled "Friends" with arrows labeled "W_friend", 1 orange node labeled "Mention" with arrow labeled "W_mention", 1 purple node labeled "Reply" with arrow labeled "W_reply". A self-loop on center node labeled "W_0 (self)". Below, the RGCN equation. Caption: "Each relation type has its OWN learned weight matrix". Clean diagram, color-coded by relation type, academic thesis style.
```

**Figure 5.3 — RGCN Model Architecture (788 → 256 → 2)**

```
A vertical neural network architecture diagram on white background. Top: wide input layer bar labeled "Input: 788-D features (20 profile + 768 LaBSE)". Arrow down to box "RGCNConv Layer 1 (788→256, 7 relations)" in blue. Arrow down to box "ReLU Activation". Arrow down to box "Dropout (p=0.5)". Arrow down to smaller box "RGCNConv Layer 2 (256→2, 7 relations)" in blue. Arrow down to narrow output bar "Output: 2-D logits". Arrow down to box "Softmax → [P(human), P(bot)]". Arrow down to result "Prediction: Bot (92%)". Progressively narrowing width from 788 to 256 to 2. Clean architecture diagram, blue layer boxes, white background.
```

**Figure 5.4 — Training Pipeline Flowchart**

```
A flowchart on white background showing the RGCN training process. Start → "Load graph_data.pt (10199 nodes, 7 relations)" → "Compute class weights (w_bot = N_human/N_bot ≈ 2.7)" → "Initialize RGCN model + Adam optimizer (lr=0.001)" → Loop start "Epoch 1 to 200": → "Forward pass on full graph" → "Compute weighted CrossEntropyLoss on train_mask" → "Backpropagate + optimizer step" → "Evaluate on val_mask" → Diamond "val_acc > best_val?" Yes → "Save best_rgcn.pt" → Loop end. After loop: "Load best model → Evaluate on test_mask → Report final metrics". Clean flowchart with loop indication, blue boxes, white background.
```

**Figure 5.5 — Inference Pipeline from Request to Prediction**

```
A horizontal pipeline diagram on white background. Left: "Request JSON {target, neighbors, relations}" → "build_mini_graph()" box containing "Filter neighbors, Build node features (788-D each), Create edge_index + edge_type" → "PyG Data object (x, edge_index, edge_type)" → "RGCN Forward Pass (no grad)" → "Extract target node logits (2-D)" → "Softmax" → "Result: {label: bot, prob: 0.92, confidence: 0.92}". Below the RGCN box: "best_rgcn.pt (pre-loaded at startup)". Clean left-to-right pipeline, blue boxes, white background, engineering thesis style.
```

---

## Chapter 6 — Use matplotlib/Excel with these specs

**Figure 6.1 — Comparative Test Accuracy Across GNN Models**

```
Bar chart. X-axis: GCN, GAT, GraphSAGE, RGCN. Y-axis: Test Accuracy (0.75 to 0.90).
Values: GCN=0.7921, GAT=0.8167, GraphSAGE=0.8716, RGCN=0.8823.
RGCN bar highlighted in darker blue. Value labels on top of each bar. Title: "Test Accuracy Comparison of GNN Architectures on MGTAB". Grid lines on Y-axis. White background, professional academic chart style.
```

**Figure 6.2 — Training Loss Curve over 200 Epochs**

```
Line chart. X-axis: Epoch (1 to 200). Y-axis: Training Loss (0.0 to 0.8).
Single blue line starting at ~0.72, dropping steeply to ~0.3 by epoch 40, then gradually decreasing to ~0.21 by epoch 200. Smooth curve. Title: "RGCN Training Loss over 200 Epochs". Grid lines. White background, academic style.
```

**Figure 6.3 — Training and Validation Accuracy Curves**

```
Line chart with two lines. X-axis: Epoch (1 to 200). Y-axis: Accuracy (0.50 to 0.95).
Blue solid line "Training Accuracy": starts ~0.55, rises steeply to ~0.85 by epoch 40, plateaus at ~0.895 by epoch 200.
Orange dashed line "Validation Accuracy": follows similar shape but plateaus ~2.5% lower at ~0.87.
Legend in top-right. Title: "RGCN Training and Validation Accuracy". Grid lines. White background.
```

**Figure 6.4 — Confusion Matrix — RGCN on Test Set**

```
2x2 heatmap confusion matrix. Rows labeled "Actual Human" and "Actual Bot". Columns labeled "Predicted Human" and "Predicted Bot". Color scheme: darker blue = higher count, lighter = lower. Cells show count numbers. Top-left (TN): large number ~1200. Top-right (FP): small number ~180. Bottom-left (FN): small number ~60. Bottom-right (TP): large number ~560. Title: "RGCN Confusion Matrix on MGTAB Test Set". Academic style with seaborn/matplotlib aesthetic.
```

**Figure 6.5 — Bot Recall Comparison Across Models**

```
Bar chart. X-axis: GCN, GAT, GraphSAGE, RGCN. Y-axis: Bot Recall (0.60 to 0.95).
Values: GCN=0.687, GAT=0.8453, GraphSAGE=0.8885, RGCN=0.9029.
RGCN bar highlighted in red/coral. Value labels on each bar. Title: "Bot Recall Comparison Across GNN Architectures". Horizontal dashed line at 0.90 labeled "90% threshold". Grid lines. White background, academic style.
```

**Figure 6.6 — Top-5 Features by Information Gain**

```
Horizontal bar chart. Y-axis (top to bottom): followers_friends_ratio, statuses_count, listed_count, description_length, favourites_count. X-axis: Relative Importance (0.0 to 1.0). Bars colored in gradient from dark blue (highest) to light blue (lowest). Title: "Top-5 Discriminative Features by Information Gain". Clean academic chart, white background, value labels at bar ends.
```

**Figure 6.7 — Screenshot of Live Detection at mgtab.me**

```
This is NOT an AI-generated image. Take an actual screenshot of the live website at https://www.mgtab.me/ showing:
1. The DetectorPage with a username entered
2. The 5-step progress stepper showing completed steps
3. The final prediction result card showing "Bot" or "Human" with confidence percentage
Use browser screenshot or Snipping Tool. Crop to show only the relevant UI area.
```

---

### Recommended Tools per Figure Type

| Type | Figures | Best Tool |
|------|---------|-----------|
| Architecture/flowcharts | 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.4, 4.5, 5.3, 5.4, 5.5 | **draw.io** (free) or Lucidchart |
| Network/graph diagrams | 1.2, 4.6, 4.7, 5.1, 5.2 | **draw.io** or yEd |
| Feature layout | 4.3 | draw.io or PowerPoint |
| Data charts | 6.1–6.6 | **matplotlib** (Python) or Excel |
| Timeline | 1.1 | draw.io, Canva, or AI image gen |
| Screenshot | 6.7 | **Browser screenshot** of mgtab.me |

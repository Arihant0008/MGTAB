# 🧠 MGTAB Bot Detector — Complete Project Guide (Part 1)
## *"Explain it like I'm a Junior Developer"*

> [!NOTE]
> This guide explains **every single piece** of the MGTAB Bot Detector — from the Twitter scraper to the neural network — in simple language with flowcharts. Use this as your go-to reference as project head.

---

## 📌 What Does This Project Do? (The 30-Second Pitch)

**Problem:** Twitter/X is full of bot accounts that spread misinformation, spam, and manipulate trends.

**Our Solution:** We built a web app where you type a Twitter username, and our system:
1. **Scrapes** that user's profile + tweets + social network from Twitter
2. **Builds a graph** (like a social network map) with that user and their connections
3. **Runs a Graph Neural Network (RGCN)** on that graph
4. **Tells you:** "This is a **Bot** (92% confidence)" or "This is a **Human** (87% confidence)"

**Why is this better than other bot detectors?** Most detectors only look at the profile. We look at the **relationships** — who follows whom, who replies to whom, who shares the same URLs — using **7 different relationship types**. This is like checking not just a person's ID card, but also who their friends are.

---

## 🏗️ System Architecture — The Big Picture

```mermaid
graph TB
    subgraph USER["🧑 User"]
        Browser["Browser (React App)"]
    end

    subgraph FRONTEND["📱 Frontend (Vite + React)"]
        HomePage["HomePage<br/>Landing Page"]
        DetectorPage["DetectorPage<br/>Main Bot Detection UI"]
        AnalyticsPage["AnalyticsPage<br/>Model Stats & Research"]
        APIClient["predict.js<br/>API Client (SSE + REST)"]
    end

    subgraph BACKEND["⚙️ Backend (FastAPI + Python)"]
        MainAPI["main.py<br/>FastAPI Endpoints"]
        Scraper["scraper.py<br/>Twitter/X Scraper (Scweet)"]
        Features["features.py<br/>Feature Engineering (788-dim)"]
        GraphBuilder["graph_builder.py<br/>PyG Graph Constructor"]
        Inference["inference.py<br/>RGCN Inference Engine"]
        RGCNModel["rgcn_model.py<br/>2-Layer RGCN"]
        Normalization["normalization.py<br/>MinMax + Log Scaling"]
        Config["config.py<br/>Constants & Settings"]
    end

    subgraph DATA["💾 Data & Model"]
        ModelFile["best_rgcn.pt<br/>Trained Model Weights"]
        LaBSE["LaBSE Model<br/>(HuggingFace, downloaded)"]
    end

    Browser --> HomePage
    Browser --> DetectorPage
    Browser --> AnalyticsPage
    DetectorPage --> APIClient
    APIClient -->|"SSE Stream"| MainAPI
    APIClient -->|"POST /predict/user"| MainAPI
    MainAPI --> Scraper
    MainAPI --> Inference
    Scraper -->|"Raw profile + tweets"| Features
    Features --> GraphBuilder
    Features -->|"Encode tweets"| LaBSE
    GraphBuilder --> Inference
    Inference --> RGCNModel
    RGCNModel --> ModelFile
    Features --> Normalization
    Config --> MainAPI
    Config --> Scraper
    Config --> Features
    Config --> GraphBuilder
```

---

## 🔄 The Complete Data Flow — What Happens When You Click "Analyze"

```mermaid
sequenceDiagram
    participant U as 🧑 User
    participant F as 📱 Frontend
    participant API as ⚙️ FastAPI
    participant S as 🕷️ Scraper
    participant FE as 🔢 Features
    participant GB as 📊 Graph Builder
    participant RGCN as 🤖 RGCN Model

    U->>F: Types "@elonmusk" → clicks Analyze
    F->>API: GET /predict/username/elonmusk (SSE)
    
    Note over API,S: Step 1: Scrape Profile
    API->>S: scrape_ego_graph("elonmusk")
    S-->>F: SSE: "Scraping profile..."
    S->>S: Fetch profile via Scweet API
    S->>S: Fetch 20 recent tweets
    
    Note over S: Step 2: Discover Network
    S-->>F: SSE: "Fetching network..."
    S->>S: Fetch followers (up to 10)
    S->>S: Fetch following (up to 10)
    S->>S: Extract mentions from tweets
    S->>S: Extract replies from tweets
    S->>S: Extract quotes from tweets
    
    Note over S: Step 3: Enrich Neighbors
    S-->>F: SSE: "Enriching neighbors..."
    S->>S: For each neighbor: fetch profile + 5 tweets
    S->>S: Track URLs & hashtags for co-occurrence
    
    Note over S,GB: Step 4: Build Graph
    S-->>F: SSE: "Building graph..."
    S->>API: Return request_data + scrape_meta
    API->>FE: Build 788-dim features for each node
    FE->>FE: 20 profile features (normalized)
    FE->>FE: 768 tweet features (LaBSE encoding)
    FE->>GB: Features → build_mini_graph()
    GB->>GB: Create PyG Data object with edges
    
    Note over RGCN: Step 5: RGCN Inference
    API-->>F: SSE: "Running RGCN..."
    GB->>RGCN: Data(x, edge_index, edge_type)
    RGCN->>RGCN: Forward pass (2 layers)
    RGCN->>API: {label: "bot", prob: 0.92}
    
    API-->>F: SSE: result event
    F->>U: Shows "🤖 Bot Detected — 92% confidence"
```

---

## 📂 File-by-File Guide — Which Files to Study

### Priority Map: "What should I read first?"

| Priority | File | What It Does | Lines | Difficulty |
|----------|------|-------------|-------|------------|
| ⭐⭐⭐ | `backend/app/main.py` | The brain — all API endpoints, SSE pipeline | 373 | Medium |
| ⭐⭐⭐ | `backend/app/scraper.py` | Twitter scraping engine (biggest file) | 809 | Hard |
| ⭐⭐⭐ | `backend/app/features.py` | 788-dim feature vector construction | 220 | Medium |
| ⭐⭐ | `backend/app/graph_builder.py` | Builds the PyG graph from scraped data | 220 | Medium |
| ⭐⭐ | `backend/app/inference.py` | Loads model + runs prediction | 105 | Easy |
| ⭐⭐ | `backend/app/rgcn_model.py` | The actual neural network (tiny!) | 39 | Easy |
| ⭐ | `backend/app/normalization.py` | Number scaling utilities | 98 | Easy |
| ⭐ | `backend/app/config.py` | All constants and settings | 80 | Easy |
| ⭐⭐ | `frontend/src/pages/DetectorPage.jsx` | Main UI with SSE stepper | 441 | Medium |
| ⭐ | `frontend/src/api/predict.js` | Frontend ↔ Backend communication | 161 | Easy |
| ⭐ | `Datasets.../6. Step - Models/rgcn_model.py` | How the model was trained | 130 | Medium |

---

## 🕷️ THE SCRAPER — `scraper.py` (Explained Simply)

### What is the Scraper?

Think of it as a **robot that goes to Twitter, logs in with your cookie, and copies information** about a user and their friends. It's like a detective gathering evidence.

### How Does Authentication Work?

```mermaid
graph LR
    A["You log into Twitter<br/>in your browser"] --> B["Open DevTools (F12)"]
    B --> C["Application → Cookies<br/>→ copy 'auth_token'"]
    C --> D["Paste into backend/.env<br/>TWITTER_AUTH_TOKEN=abc123"]
    D --> E["Scweet library uses<br/>this cookie to act<br/>as 'you' on Twitter"]
```

**Simple explanation:** Instead of a username/password, we use a **cookie** from your browser. It's like borrowing your VIP badge to get into the Twitter building.

### The Ego-Graph Scraping Pipeline (The Big Function)

The main function is `scrape_ego_graph()`. Here's what it does step by step:

```mermaid
flowchart TD
    START["scrape_ego_graph('elonmusk')"] --> CLEAN["Clean username<br/>Remove @ symbol"]
    CLEAN --> AUTH["Ensure Scweet client<br/>is authenticated"]
    
    AUTH --> S1["📌 STEP 1: Scrape Target"]
    S1 --> PROFILE["Fetch profile via<br/>aget_user_info(['elonmusk'])"]
    PROFILE --> CHECK["Is account private?"]
    CHECK -->|Yes| ERROR["❌ Throw error:<br/>Cannot scrape private accounts"]
    CHECK -->|No| TWEETS["Fetch 20 recent tweets<br/>via aget_profile_tweets()"]
    TWEETS --> EXTRACT["Extract from tweets:<br/>• Tweet text<br/>• URLs shared<br/>• Hashtags used"]
    
    EXTRACT --> S2["📌 STEP 2: Discover Neighbors"]
    S2 --> FOL["Fetch up to 10 followers"]
    FOL --> FRI["Fetch up to 10 following"]
    FRI --> MEN["Extract @mentions from tweets"]
    MEN --> REP["Extract reply usernames from tweets"]
    REP --> QUO["Extract quoted usernames from tweets"]
    QUO --> DEDUP["Deduplicate all neighbors<br/>~50 unique max"]
    
    DEDUP --> S3["📌 STEP 3: Enrich Neighbors"]
    S3 --> LOOP["For each neighbor:"]
    LOOP --> NPROF["Fetch their profile"]
    NPROF --> NTWEET["Fetch their 5 tweets<br/>(rate-limit resilient!)"]
    NTWEET --> NURLS["Track their URLs & hashtags"]
    NURLS --> LOOP2{"More neighbors?"}
    LOOP2 -->|Yes| LOOP
    LOOP2 -->|No| S4
    
    S4["📌 STEP 4: Build Relations"] --> EDGES["Create edge list:<br/>follower, friend, mention,<br/>reply, quoted"]
    EDGES --> URL_EDGES["Add URL co-occurrence edges<br/>(if target & neighbor share URLs)"]
    URL_EDGES --> HASH_EDGES["Add hashtag co-occurrence edges<br/>(if they share hashtags)"]
    HASH_EDGES --> RETURN["Return request_data +<br/>scrape_meta"]
```

### Key Functions in scraper.py

| Function | What It Does | Simple Analogy |
|----------|-------------|----------------|
| `scweet_user_to_profile()` | Converts Scweet's data format → our 20-field format | Translator between two languages |
| `_extract_tweet_texts()` | Pulls just the text from tweet objects | Reading only the message, ignoring metadata |
| `_extract_urls_from_tweets()` | Finds all URLs in tweets | Finding all links in a WhatsApp chat |
| `_extract_hashtags_from_tweets()` | Finds all #hashtags | Finding all topics being discussed |
| `_extract_mentions_from_tweets()` | Finds all @mentions | Finding who's being talked to |
| `_extract_reply_usernames()` | Finds who was replied to | Finding who started the conversation |
| `_extract_quoted_usernames()` | Finds whose tweets were quoted | Finding who was being referenced |
| `_is_rate_limit_error()` | Checks if Twitter said "slow down!" | Checking if the bouncer kicked you out |

### Rate-Limit Resilience — Why Tweets Can Be Empty

```mermaid
graph TD
    FETCH["Try to fetch tweets<br/>for neighbor @john"] --> RL{"Rate Limited?<br/>(HTTP 429)"}
    RL -->|No| SUCCESS["✅ Got 5 tweets<br/>Node has full data"]
    RL -->|Yes| WARN["⚠️ Log warning"]
    WARN --> EMPTY["Return empty tweet list"]
    EMPTY --> STILL["Node STILL enters graph<br/>with profile features only<br/>(tweets = zero vector)"]
    
    style SUCCESS fill:#10b981,color:#fff
    style STILL fill:#f59e0b,color:#fff
```

**Simple explanation:** If Twitter says "you're asking too fast!", we don't crash. We just use whatever profile data we have. The person still gets added to the graph, just without their tweet content.

---

## 🔢 FEATURE ENGINEERING — `features.py` (The 788-Dimension Vector)

### What is a Feature Vector?

Imagine you're describing a person using only numbers. Instead of saying "John has lots of followers, few friends, and his tweets talk about tech", you convert ALL of that into a list of 788 numbers. The AI reads these numbers to decide: bot or human.

### The 788-Dimension Breakdown

```mermaid
graph LR
    subgraph PROFILE["20 Profile Features"]
        BOOL["10 Boolean (0 or 1)<br/>verified? default_profile?<br/>geo_enabled? has_url?<br/>etc."]
        NUM["5 Numerical (scaled 0→1)<br/>followers, friends,<br/>listed, statuses, likes"]
        DER["5 Derived<br/>account_age, name_length,<br/>screen_name_length,<br/>description_length,<br/>followers/friends ratio"]
    end
    
    subgraph TWEET["768 Tweet Features"]
        LABSE["LaBSE Embedding<br/>All tweets → 768 numbers<br/>representing MEANING"]
    end
    
    PROFILE --> CONCAT["Concatenate"]
    TWEET --> CONCAT
    CONCAT --> VECTOR["Final: 788-dim vector<br/>[0.0, 1.0, 0.0, 0.45, ..., 0.12, -0.3, ...]"]
```

### How Profile Features Are Built

```mermaid
flowchart TD
    INPUT["Raw Profile Data<br/>{followers: 1500, verified: true, ...}"] --> SPLIT["Split into 3 types"]
    
    SPLIT --> BOOLS["BOOLEANS (10 features)<br/>verified → 1.0<br/>default_profile → 0.0<br/>geo_enabled → 0.0<br/>..."]
    
    SPLIT --> NUMS["NUMERICALS (5 features)<br/>followers_count: 1500<br/>↓ log(1 + 1500) = 7.31<br/>↓ MinMax scale = 0.286"]
    
    SPLIT --> DERIVED["DERIVED (5 features)<br/>screen_name length: 10 → 0.583<br/>name length: 8 → 0.143<br/>description length: 45 → 0.221<br/>account age: timestamp → log → scale<br/>followers/friends ratio: 1500/200=7.5 → scale"]
    
    BOOLS --> MERGE["Merge into 20-dim array<br/>[0,1,2,...,19] → exact positions"]
    NUMS --> MERGE
    DERIVED --> MERGE
```

### How Tweet Features Work (LaBSE)

```mermaid
flowchart TD
    TWEETS["User's Tweets<br/>['Great weather!',<br/>'Just read a book',<br/>'Love coffee ☕']"] --> TOKENIZE["LaBSE Tokenizer<br/>Convert text → numbers<br/>(tokens)"]
    
    TOKENIZE --> MODEL["LaBSE Model<br/>(Pre-trained by Google)<br/>Understands 109 languages"]
    
    MODEL --> EMBED["Each tweet → 768 numbers<br/>Tweet 1: [0.12, -0.3, 0.8, ...]<br/>Tweet 2: [0.05, 0.2, -0.1, ...]<br/>Tweet 3: [0.3, -0.5, 0.4, ...]"]
    
    EMBED --> SUM["SUM all embeddings<br/>(NOT average!)<br/>[0.47, -0.6, 1.1, ...]"]
    
    SUM --> FINAL["768-dim tweet vector<br/>Captures the MEANING<br/>of all tweets combined"]
    
    style SUM fill:#f59e0b,color:#fff
```

> [!IMPORTANT]
> **Why SUM and not AVERAGE?** The original MGTAB training data used summed embeddings (giving norms ~18-20). If we averaged, the numbers would be too small (~5-6) and the trained model would ignore tweet features. We must match the training data format exactly.

---

## 📊 GRAPH BUILDER — `graph_builder.py` (Building the Social Network Map)

### What Is a Graph?

A **graph** is just dots (nodes) connected by lines (edges). In our case:
- **Nodes** = Twitter users (the target + their neighbors)
- **Edges** = Relationships between them (follows, mentions, replies, etc.)

### The 7 Relationship Types

```mermaid
graph LR
    subgraph DIRECTED["5 Directed Relations"]
        R0["R0: Follower<br/>B follows A → B→A"]
        R1["R1: Friend<br/>A follows B → A→B"]
        R2["R2: Mention<br/>A mentions B → A→B"]
        R3["R3: Reply<br/>A replies to B → A→B"]
        R4["R4: Quoted<br/>A quotes B → A→B"]
    end
    
    subgraph UNDIRECTED["2 Undirected Relations"]
        R5["R5: URL<br/>Both share same URL → A↔B"]
        R6["R6: Hashtag<br/>Both use same hashtag → A↔B"]
    end
```

### How `build_mini_graph()` Works

```mermaid
flowchart TD
    INPUT["Request Data<br/>{target, neighbors[], relations[]}"] --> FILTER_N["Filter neighbors:<br/>Keep only those with<br/>REAL profile/tweet data"]
    
    FILTER_N --> FILTER_R["Filter relations:<br/>Keep only edges where<br/>neighbor has real data"]
    
    FILTER_R --> WHY["⚠️ WHY FILTER?<br/>Zero-vector neighbors<br/>corrupt the model's<br/>mean aggregation"]
    
    FILTER_R --> NODES["Build node index<br/>node 0 = target<br/>node 1 = neighbor_A<br/>node 2 = neighbor_B<br/>..."]
    
    NODES --> FEATURES["Build feature matrix<br/>Node 0: build_node_feature(target) → 788-dim<br/>Node 1: build_node_feature(neighbor_A) → 788-dim<br/>..."]
    
    FEATURES --> EDGES["Build edge lists with<br/>correct directions:<br/>follower: neighbor→target<br/>friend: target→neighbor<br/>url/hashtag: both directions"]
    
    EDGES --> SELFLOOP{"Any edges exist?"}
    SELFLOOP -->|No| ADDSELF["Add self-loop<br/>(target→target, type 0)<br/>So RGCN can still run"]
    SELFLOOP -->|Yes| ASSEMBLE["Assemble PyG Data"]
    ADDSELF --> ASSEMBLE
    
    ASSEMBLE --> OUTPUT["Data(<br/>  x = feature matrix,<br/>  edge_index = connections,<br/>  edge_type = relation IDs<br/>)"]
```

### Example Mini-Graph

```mermaid
graph TD
    TARGET["🎯 Target: @elonmusk<br/>(Node 0, 788-dim features)"]
    
    N1["👤 @user1<br/>(Node 1)"]
    N2["👤 @user2<br/>(Node 2)"]
    N3["👤 @user3<br/>(Node 3)"]
    
    N1 -->|"R0: follower"| TARGET
    TARGET -->|"R1: friend"| N2
    TARGET -->|"R2: mention"| N3
    TARGET -->|"R3: reply"| N1
    TARGET <-->|"R6: hashtag"| N2
    
    style TARGET fill:#3b82f6,color:#fff
```

---

## 🤖 THE RGCN MODEL — `rgcn_model.py` (The AI Brain)

### What is RGCN?

**RGCN = Relational Graph Convolutional Network**. Think of it like this:

- A regular neural network reads a flat list of numbers
- A **Graph** neural network reads a network of connected nodes
- An **R**GCN reads a network where **each connection has a TYPE** (follower, friend, mention, etc.)

### The Architecture (Only 39 Lines!)

```mermaid
flowchart TD
    INPUT["Input: 788-dim features<br/>for each node"] --> CONV1["Layer 1: RGCNConv<br/>788 → 256 dimensions<br/>(7 relation types)"]
    
    CONV1 --> RELU["ReLU Activation<br/>Remove negative values"]
    
    RELU --> DROP["Dropout (50%)<br/>Randomly turn off neurons<br/>during training to prevent<br/>overfitting"]
    
    DROP --> CONV2["Layer 2: RGCNConv<br/>256 → 2 dimensions<br/>(7 relation types)"]
    
    CONV2 --> OUTPUT["Output: 2 numbers<br/>[score_human, score_bot]"]
    
    OUTPUT --> SOFTMAX["Softmax<br/>Convert to probabilities<br/>[0.08, 0.92]"]
    
    SOFTMAX --> RESULT["Result:<br/>Bot (92% confidence)"]
```

### How RGCNConv Works (The Magic)

```mermaid
flowchart TD
    NODE["Target node<br/>features: 788-dim"] --> SELF["Self-transform:<br/>W_self × features"]
    
    NODE --> AGG["For each relation type:"]
    AGG --> R0["R0 (follower) neighbors:<br/>W_follower × mean(neighbor_features)"]
    AGG --> R1["R1 (friend) neighbors:<br/>W_friend × mean(neighbor_features)"]
    AGG --> R2["R2 (mention) neighbors:<br/>W_mention × mean(neighbor_features)"]
    AGG --> DOTS["... (7 total)"]
    
    SELF --> SUM["SUM everything together"]
    R0 --> SUM
    R1 --> SUM
    R2 --> SUM
    DOTS --> SUM
    
    SUM --> NEW["New node representation<br/>(256-dim after Layer 1)"]
```

**Simple explanation:** The RGCN looks at each node and asks: *"What do your followers look like? What do your friends look like? Who mentions you?"* — and combines all those answers using **different learned weights for each relationship type**.

---

## ⚡ INFERENCE ENGINE — `inference.py` (Running the Model)

```mermaid
flowchart LR
    subgraph ENGINE["InferenceEngine (Singleton)"]
        LOAD["__init__:<br/>Load best_rgcn.pt<br/>Set to eval mode<br/>Put on CPU"]
        
        PREDICT["predict():<br/>1. Move data to device<br/>2. Forward pass (no grad)<br/>3. Get target node logits<br/>4. Softmax → probabilities<br/>5. Return label + confidence"]
        
        FULL["predict_from_request():<br/>1. Call build_mini_graph()<br/>2. Call predict()<br/>3. Add graph metadata"]
    end
    
    REQUEST["Request JSON"] --> FULL
    FULL --> RESULT["{'label': 'bot',<br/>'prob_human': 0.08,<br/>'prob_bot': 0.92,<br/>'confidence': 0.92}"]
```

---

*Continued in Part 2 → Frontend, API Endpoints, Training Pipeline, and Normalization...*
